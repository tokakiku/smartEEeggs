import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from urllib.parse import urlparse

"""Mongo 知识库服务。

提供四层文档的统一写入、检索、主次优先级维护与统计接口。
"""


class MongoKBError(RuntimeError):
    """Mongo 知识库存储异常基类。"""
    pass


class MongoKBUnavailableError(MongoKBError):
    """Mongo 不可用时抛出，供 API/业务层返回清晰错误。"""
    pass


def _is_true(value: Optional[str], default: bool) -> bool:
    """环境变量布尔解析工具。"""
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class MongoKBService:
    """MongoDB 知识库存储服务：四层统一 upsert 与查询。"""

    DB_NAME_DEFAULT = "ruijie_kb"
    URI_DEFAULT = "mongodb://localhost:27017"

    COLLECTION_MAP = {
        "syllabus": "syllabus_docs",
        "textbook": "textbook_docs",
        "resource": "resource_docs",
        "hotspot": "hotspot_docs",
    }
    SYLLABUS_PRIORITY_FIELDS = [
        "is_primary",
        "course_name",
        "course_code",
        "school",
        "department",
        "major",
        "academic_year",
        "semester",
        "version",
        "teacher",
        "effective_date",
        "priority_score",
    ]
    TEXTBOOK_PRIORITY_FIELDS = [
        "textbook_role",
        "subject",
        "title",
        "edition",
        "academic_year",
        "author",
        "authors",
        "source_file",
        "priority_score",
    ]

    def __init__(
        self,
        uri: Optional[str] = None,
        db_name: Optional[str] = None,
        enabled: Optional[bool] = None,
        required: Optional[bool] = None,
        timeout_ms: Optional[int] = None,
        use_mock: Optional[bool] = None,
        client: Any = None,
    ) -> None:
        self.uri = uri or os.getenv("MONGODB_URI", self.URI_DEFAULT)
        self.db_name = db_name or os.getenv("MONGODB_DB_NAME", self.DB_NAME_DEFAULT)
        self.enabled = enabled if enabled is not None else _is_true(os.getenv("MONGODB_ENABLED"), True)
        self.required = required if required is not None else _is_true(os.getenv("MONGODB_REQUIRED"), False)
        self.timeout_ms = timeout_ms or int(os.getenv("MONGODB_TIMEOUT_MS", "3000"))
        self.use_mock = use_mock if use_mock is not None else _is_true(os.getenv("MONGODB_USE_MOCK"), False)

        self.client = client
        self.db = None
        self.unavailable_reason: Optional[str] = None
        self.backend = "mock" if self.use_mock else "mongo"

        if self.enabled:
            self._connect()

    @classmethod
    def from_env(cls) -> "MongoKBService":
        """按环境变量初始化，便于部署时注入连接参数。"""
        return cls()

    @property
    def is_available(self) -> bool:
        """当前 Mongo 服务是否可用。"""
        return self.enabled and self.db is not None

    def _connect(self) -> None:
        """建立 Mongo 连接并创建索引。"""
        try:
            if self.client is None:
                if self.use_mock:
                    import mongomock

                    self.client = mongomock.MongoClient()
                else:
                    from pymongo import MongoClient

                    self.client = MongoClient(self.uri, serverSelectionTimeoutMS=self.timeout_ms)

            if not self.use_mock:
                self.client.admin.command("ping")
            self.db = self.client[self.db_name]
            self._ensure_indexes()
        except Exception as exc:
            self.unavailable_reason = str(exc)
            self.db = None
            if self.required:
                raise MongoKBUnavailableError(f"mongodb unavailable: {exc}") from exc

    def _ensure_available(self) -> None:
        """统一可用性检查。"""
        if not self.enabled:
            raise MongoKBUnavailableError("mongodb disabled by config")
        if self.db is None:
            reason = self.unavailable_reason or "mongodb not connected"
            raise MongoKBUnavailableError(reason)

    def _ensure_indexes(self) -> None:
        """建立最小必要索引与唯一约束。"""
        try:
            from pymongo import ASCENDING
        except Exception:
            # 单测使用 mongomock 且未安装 pymongo 时，降级为常量 1。
            ASCENDING = 1

        for layer, collection_name in self.COLLECTION_MAP.items():
            collection = self.db[collection_name]
            collection.create_index(
                [("layer", ASCENDING), ("source_file", ASCENDING)],
                unique=True,
                name="uniq_layer_source_file",
            )
            collection.create_index([("doc_id", ASCENDING)], name="idx_doc_id")
            collection.create_index([("title", ASCENDING)], name="idx_title")
            collection.create_index([("subject", ASCENDING)], name="idx_subject")
            collection.create_index([("knowledge_points", ASCENDING)], name="idx_knowledge_points")
            collection.create_index([("page_roles", ASCENDING)], name="idx_page_roles")
            collection.create_index([("event_types", ASCENDING)], name="idx_event_types")
            collection.create_index([("updated_at", ASCENDING)], name="idx_updated_at")
            if layer == "syllabus":
                collection.create_index([("syllabus_meta.is_primary", ASCENDING)], name="idx_syllabus_primary")
                collection.create_index([("syllabus_meta.course_name", ASCENDING)], name="idx_syllabus_course_name")
                collection.create_index([("syllabus_meta.course_code", ASCENDING)], name="idx_syllabus_course_code")
                collection.create_index(
                    [("syllabus_meta.priority_score", ASCENDING)],
                    name="idx_syllabus_priority_score",
                )
            if layer == "textbook":
                collection.create_index([("textbook_meta.textbook_role", ASCENDING)], name="idx_textbook_role")
                collection.create_index([("textbook_meta.subject", ASCENDING)], name="idx_textbook_subject")
                collection.create_index([("textbook_meta.priority_score", ASCENDING)], name="idx_textbook_priority")

    def save_extraction_result(self, layer: str, payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        """写入或更新 extractor 结果到对应 layer collection。"""
        self._ensure_available()
        normalized_layer = self._normalize_layer(layer)
        collection = self.db[self.COLLECTION_MAP[normalized_layer]]

        source_file = str(metadata.get("source_file") or "")
        doc_id = str(metadata.get("doc_id") or "")
        if not source_file:
            source_file = doc_id
        if not source_file:
            raise MongoKBError("source_file and doc_id are both empty")

        parser_name = str(metadata.get("parser_name") or "")
        source_type = metadata.get("source_type")
        title = self._extract_title(normalized_layer, payload)
        if not title:
            title = source_file
        subject = self._extract_subject(normalized_layer, payload)
        syllabus_meta: Dict[str, Any] = {}
        textbook_meta: Dict[str, Any] = {}
        if normalized_layer == "syllabus":
            syllabus_meta = self._extract_syllabus_priority_meta(payload=payload, metadata=metadata)
        if normalized_layer == "textbook":
            textbook_meta = self._extract_textbook_priority_meta(
                payload=payload,
                metadata=metadata,
                source_file=source_file,
                title=title,
                subject=subject,
            )

        now_iso = datetime.now(timezone.utc).isoformat()
        key = {"layer": normalized_layer, "source_file": source_file}
        set_payload = {
            "_id": f"{normalized_layer}:{source_file}",
            "layer": normalized_layer,
            "doc_id": doc_id,
            "source_file": source_file,
            "source_type": source_type,
            "parser_name": parser_name,
            "title": title,
            "subject": subject,
            "updated_at": now_iso,
            "status": "active",
            "data": payload,
            "knowledge_points": self._extract_knowledge_points(normalized_layer, payload),
            "page_roles": self._extract_page_roles(normalized_layer, payload),
            "event_types": self._extract_event_types(normalized_layer, payload),
        }
        if syllabus_meta:
            set_payload["syllabus_meta"] = syllabus_meta
        if textbook_meta:
            set_payload["textbook_meta"] = textbook_meta
        set_on_insert = {"created_at": now_iso}

        result = collection.update_one(key, {"$set": set_payload, "$setOnInsert": set_on_insert}, upsert=True)
        operation = "upserted" if result.upserted_id is not None else "updated"
        return {
            "status": operation,
            "collection": self.COLLECTION_MAP[normalized_layer],
            "backend": self.backend,
            "matched_count": result.matched_count,
            "modified_count": result.modified_count,
            "upserted_id": str(result.upserted_id) if result.upserted_id is not None else None,
            "doc_id": doc_id,
            "source_file": source_file,
        }

    def search_documents(
        self,
        layer: Optional[str] = None,
        source_file: Optional[str] = None,
        title: Optional[str] = None,
        subject: Optional[str] = None,
        knowledge_point: Optional[str] = None,
        page_role: Optional[str] = None,
        event_type: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """基础检索：支持 layer/source/title/subject/kp/page_role/event_type。"""
        self._ensure_available()
        limit = max(1, min(limit, 100))

        query: Dict[str, Any] = {}
        if source_file:
            query["source_file"] = {"$regex": re.escape(source_file), "$options": "i"}
        if title:
            query["title"] = {"$regex": re.escape(title), "$options": "i"}
        if subject:
            query["subject"] = {"$regex": re.escape(subject), "$options": "i"}
        if knowledge_point:
            query["knowledge_points"] = {"$regex": re.escape(knowledge_point), "$options": "i"}
        if page_role:
            query["page_roles"] = {"$regex": re.escape(page_role), "$options": "i"}
        if event_type:
            query["event_types"] = {"$regex": re.escape(event_type), "$options": "i"}

        results: List[Dict[str, Any]] = []
        target_layers = [self._normalize_layer(layer)] if layer else list(self.COLLECTION_MAP.keys())
        for layer_name in target_layers:
            collection_name = self.COLLECTION_MAP[layer_name]
            collection = self.db[collection_name]
            docs = list(collection.find(query, {"data": 0}).sort("updated_at", -1).limit(limit))
            for doc in docs:
                results.append(self._serialize_doc(doc, collection_name))

        results.sort(key=self._search_sort_key, reverse=True)
        return results[:limit]

    def get_document(self, layer: str, doc_id: str) -> Optional[Dict[str, Any]]:
        """按 layer + doc_id 获取完整结构化文档。"""
        self._ensure_available()
        normalized_layer = self._normalize_layer(layer)
        collection = self.db[self.COLLECTION_MAP[normalized_layer]]
        doc = collection.find_one({"layer": normalized_layer, "doc_id": doc_id})
        if not doc:
            return None
        return self._serialize_doc(doc, self.COLLECTION_MAP[normalized_layer])

    def set_primary_syllabus(
        self,
        doc_id: str,
        course_name: Optional[str] = None,
        course_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 将指定 syllabus 设为 primary，并取消同课程的其他 primary。
        self._ensure_available()
        collection = self.db[self.COLLECTION_MAP["syllabus"]]
        target = collection.find_one(
            {
                "layer": "syllabus",
                "doc_id": doc_id,
                "$or": [{"status": "active"}, {"status": {"$exists": False}}],
            }
        )
        if not target:
            raise MongoKBError(f"syllabus not found: {doc_id}")

        target_meta = self._normalize_syllabus_meta(target.get("syllabus_meta") or {})
        match_course_name = (course_name or target_meta.get("course_name") or target.get("title") or "").strip()
        match_course_code = (course_code or target_meta.get("course_code") or "").strip()

        unset_query: Dict[str, Any] = {
            "layer": "syllabus",
            "$or": [{"status": "active"}, {"status": {"$exists": False}}],
        }
        if match_course_code:
            unset_query["syllabus_meta.course_code"] = match_course_code
        elif match_course_name:
            unset_query["syllabus_meta.course_name"] = match_course_name

        now_iso = datetime.now(timezone.utc).isoformat()
        unset_result = collection.update_many(
            unset_query,
            {"$set": {"syllabus_meta.is_primary": False, "updated_at": now_iso}},
        )

        current_meta = self._normalize_syllabus_meta(target.get("syllabus_meta") or {})
        current_meta["is_primary"] = True
        if match_course_name and not current_meta.get("course_name"):
            current_meta["course_name"] = match_course_name
        if match_course_code and not current_meta.get("course_code"):
            current_meta["course_code"] = match_course_code
        current_meta["priority_score"] = self._compute_syllabus_priority_score(current_meta)

        update_result = collection.update_one(
            {"layer": "syllabus", "doc_id": doc_id},
            {"$set": {"syllabus_meta": current_meta, "updated_at": now_iso}},
        )

        return {
            "status": "ok",
            "doc_id": doc_id,
            "course_name": current_meta.get("course_name"),
            "course_code": current_meta.get("course_code"),
            "unset_count": int(unset_result.modified_count),
            "updated_count": int(update_result.modified_count),
            "syllabus_meta": current_meta,
        }

    def set_primary_textbook(
        self,
        doc_id: str,
        subject: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Dict[str, Any]:
        # 将指定 textbook 标记为 main，并将同主题教材降为 supplementary。
        self._ensure_available()
        collection = self.db[self.COLLECTION_MAP["textbook"]]
        target = collection.find_one(
            {
                "layer": "textbook",
                "doc_id": doc_id,
                "$or": [{"status": "active"}, {"status": {"$exists": False}}],
            }
        )
        if not target:
            raise MongoKBError(f"textbook not found: {doc_id}")

        target_meta = self._normalize_textbook_meta(target.get("textbook_meta") or {})
        match_subject = (subject or target_meta.get("subject") or target.get("subject") or "").strip()
        match_title = (title or target_meta.get("title") or target.get("title") or "").strip()

        unset_query: Dict[str, Any] = {
            "layer": "textbook",
            "$or": [{"status": "active"}, {"status": {"$exists": False}}],
        }
        scope_filter: Dict[str, Any] = {}
        if match_subject:
            scope_filter["$or"] = [
                {"textbook_meta.subject": match_subject},
                {"subject": match_subject},
            ]
        elif match_title:
            scope_filter["$or"] = [
                {"textbook_meta.title": match_title},
                {"title": match_title},
            ]
        if scope_filter:
            unset_query = {"$and": [unset_query, scope_filter]}

        now_iso = datetime.now(timezone.utc).isoformat()
        unset_result = collection.update_many(
            unset_query,
            {
                "$set": {
                    "textbook_meta.textbook_role": "supplementary",
                    "updated_at": now_iso,
                }
            },
        )
        # 批量降级后补齐每条教材 priority_score，避免排序分数滞后。
        for doc in collection.find(unset_query):
            meta = self._normalize_textbook_meta(doc.get("textbook_meta") or {})
            meta["textbook_role"] = "supplementary"
            meta["priority_score"] = self._compute_textbook_priority_score(meta)
            collection.update_one(
                {"_id": doc.get("_id")},
                {"$set": {"textbook_meta": meta, "updated_at": now_iso}},
            )

        current_meta = self._normalize_textbook_meta(target.get("textbook_meta") or {})
        current_meta["textbook_role"] = "main"
        if match_subject and not current_meta.get("subject"):
            current_meta["subject"] = match_subject
        if match_title and not current_meta.get("title"):
            current_meta["title"] = match_title
        current_meta["source_file"] = current_meta.get("source_file") or target.get("source_file")
        current_meta["priority_score"] = self._compute_textbook_priority_score(current_meta)

        update_result = collection.update_one(
            {"layer": "textbook", "doc_id": doc_id},
            {"$set": {"textbook_meta": current_meta, "updated_at": now_iso}},
        )

        return {
            "status": "ok",
            "doc_id": doc_id,
            "subject": current_meta.get("subject"),
            "title": current_meta.get("title"),
            "unset_count": int(unset_result.modified_count),
            "updated_count": int(update_result.modified_count),
            "textbook_meta": current_meta,
        }

    def get_collection_counts(self) -> Dict[str, int]:
        """返回各 collection 当前条数。"""
        self._ensure_available()
        counts: Dict[str, int] = {}
        for layer, collection_name in self.COLLECTION_MAP.items():
            counts[layer] = int(self.db[collection_name].count_documents({}))
        return counts

    def get_active_collection_counts(self) -> Dict[str, int]:
        """仅统计 active（或未显式标注 status）的文档数。"""
        self._ensure_available()
        counts: Dict[str, int] = {}
        active_query = {"$or": [{"status": "active"}, {"status": {"$exists": False}}]}
        for layer, collection_name in self.COLLECTION_MAP.items():
            query = {"layer": layer, **active_query}
            counts[layer] = int(self.db[collection_name].count_documents(query))
        return counts

    def iter_active_documents(
        self,
        layers: Optional[List[str]] = None,
        projection: Optional[Dict[str, int]] = None,
        batch_size: int = 200,
    ):
        """迭代 active 文档，供向量化管线使用。"""
        self._ensure_available()
        target_layers = [self._normalize_layer(layer) for layer in (layers or list(self.COLLECTION_MAP.keys()))]
        active_query = {"$or": [{"status": "active"}, {"status": {"$exists": False}}]}
        for layer in target_layers:
            collection = self.db[self.COLLECTION_MAP[layer]]
            query = {"layer": layer, **active_query}
            cursor = collection.find(query, projection=projection).batch_size(max(1, int(batch_size)))
            for doc in cursor:
                yield self._serialize_doc(doc, self.COLLECTION_MAP[layer])

    def get_document_by_source_file(self, layer: str, source_file: str) -> Optional[Dict[str, Any]]:
        """按 layer + source_file 获取完整文档（用于向量结果回溯）。"""
        self._ensure_available()
        normalized_layer = self._normalize_layer(layer)
        collection = self.db[self.COLLECTION_MAP[normalized_layer]]
        doc = collection.find_one({"layer": normalized_layer, "source_file": source_file})
        if not doc:
            return None
        return self._serialize_doc(doc, self.COLLECTION_MAP[normalized_layer])

    def _normalize_layer(self, layer: Optional[str]) -> str:
        normalized = (layer or "").strip().lower()
        if normalized not in self.COLLECTION_MAP:
            raise ValueError(f"unsupported layer: {layer}")
        return normalized

    def _serialize_doc(self, doc: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """Mongo 文档序列化：ObjectId -> str，并附带 collection 名称。"""
        result = dict(doc)
        if "_id" in result:
            result["_id"] = str(result["_id"])
        result["collection"] = collection_name
        return result

    def _search_sort_key(self, item: Dict[str, Any]) -> tuple:
        collection = str(item.get("collection") or "")
        if collection == self.COLLECTION_MAP["syllabus"]:
            meta = self._normalize_syllabus_meta(item.get("syllabus_meta") or {})
            return (
                1 if meta.get("is_primary") else 0,
                float(meta.get("priority_score") or 0.0),
                item.get("updated_at") or "",
            )
        if collection == self.COLLECTION_MAP["textbook"]:
            meta = self._normalize_textbook_meta(item.get("textbook_meta") or {})
            role = str(meta.get("textbook_role") or "").strip().lower()
            role_rank = 2 if role == "main" else (0 if role == "supplementary" else 1)
            return (
                role_rank,
                float(meta.get("priority_score") or 0.0),
                item.get("updated_at") or "",
            )
        return (0, 0.0, item.get("updated_at") or "")

    def _extract_title(self, layer: str, payload: Dict[str, Any]) -> Optional[str]:
        if layer == "syllabus":
            info = payload.get("course_info") or {}
            return payload.get("course_name") or info.get("course_name")
        if layer == "textbook":
            info = payload.get("textbook_info") or {}
            title = info.get("book_title")
            if title and not self._is_directory_like_title(title):
                return title
            return None
        if layer == "resource":
            info = payload.get("resource_info") or {}
            return info.get("title")
        if layer == "hotspot":
            info = payload.get("hotspot_info") or {}
            return info.get("title")
        return None

    def _extract_subject(self, layer: str, payload: Dict[str, Any]) -> Optional[str]:
        if layer == "syllabus":
            major = str(payload.get("major") or "").strip()
            if major:
                return major
            info = payload.get("course_info") or {}
            majors = info.get("applicable_major") or []
            if not majors:
                majors = payload.get("target_major") or []
            if isinstance(majors, list) and majors:
                return " / ".join([str(item) for item in majors if item])
            return None
        if layer == "textbook":
            info = payload.get("textbook_info") or {}
            return info.get("subject")
        if layer == "resource":
            info = payload.get("resource_info") or {}
            return info.get("subject") or info.get("course_topic")
        return None

    def _extract_knowledge_points(self, layer: str, payload: Dict[str, Any]) -> List[str]:
        points: List[str] = []
        if layer == "syllabus":
            points.extend(payload.get("knowledge_points") or [])
            points.extend(payload.get("teaching_key_points") or [])
            points.extend(payload.get("teaching_difficult_points") or [])
            for module in payload.get("course_modules") or []:
                points.extend(module.get("key_points") or [])
                points.extend(module.get("difficult_points") or [])
                points.extend(module.get("learning_requirements") or [])
                module_name = module.get("module_name")
                if module_name:
                    points.append(module_name)
        elif layer == "textbook":
            for item in payload.get("knowledge_points") or []:
                name = item.get("name")
                if name:
                    points.append(name)
        elif layer == "resource":
            for page in payload.get("pages") or []:
                points.extend(page.get("knowledge_points") or [])
        elif layer == "hotspot":
            for item in payload.get("hotspot_item") or []:
                points.extend(item.get("related_knowledge_points") or [])
        return self._deduplicate(points)[:100]

    def _extract_page_roles(self, layer: str, payload: Dict[str, Any]) -> List[str]:
        if layer != "resource":
            return []
        roles: List[str] = []
        for page in payload.get("pages") or []:
            role = page.get("page_role")
            if role:
                roles.append(str(role))
        return self._deduplicate(roles)

    def _extract_event_types(self, layer: str, payload: Dict[str, Any]) -> List[str]:
        if layer != "hotspot":
            return []
        event_types: List[str] = []
        for item in payload.get("hotspot_item") or []:
            event_type = item.get("event_type")
            if event_type:
                event_types.append(str(event_type))
        return self._deduplicate(event_types)

    def _extract_syllabus_priority_meta(self, payload: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if not isinstance(metadata, dict):
            metadata = {}

        meta_input = metadata.get("syllabus_meta")
        base_meta = dict(meta_input) if isinstance(meta_input, dict) else {}
        course_info = payload.get("course_info") or {}

        normalized = self._normalize_syllabus_meta(
            {
                "is_primary": base_meta.get("is_primary", payload.get("is_primary", False)),
                "course_name": base_meta.get("course_name")
                or payload.get("course_name")
                or course_info.get("course_name"),
                "course_code": base_meta.get("course_code")
                or payload.get("course_code")
                or course_info.get("course_code"),
                "school": base_meta.get("school") or payload.get("school"),
                "department": base_meta.get("department")
                or payload.get("department")
                or course_info.get("offering_institute"),
                "major": base_meta.get("major")
                or payload.get("major")
                or self._first_list_item(payload.get("target_major"))
                or self._first_list_item(course_info.get("applicable_major")),
                "academic_year": base_meta.get("academic_year") or payload.get("academic_year"),
                "semester": base_meta.get("semester")
                or payload.get("semester")
                or course_info.get("suggested_term"),
                "version": base_meta.get("version") or payload.get("version"),
                "teacher": base_meta.get("teacher") or payload.get("teacher"),
                "effective_date": base_meta.get("effective_date") or payload.get("effective_date"),
                "priority_score": base_meta.get("priority_score", payload.get("priority_score", 0.0)),
            }
        )
        normalized["priority_score"] = self._compute_syllabus_priority_score(normalized)
        return normalized

    def _extract_textbook_priority_meta(
        self,
        payload: Dict[str, Any],
        metadata: Dict[str, Any],
        source_file: str,
        title: Optional[str],
        subject: Optional[str],
    ) -> Dict[str, Any]:
        if not isinstance(payload, dict):
            return {}
        if not isinstance(metadata, dict):
            metadata = {}

        info = payload.get("textbook_info") or {}
        meta_input = metadata.get("textbook_meta")
        base_meta = dict(meta_input) if isinstance(meta_input, dict) else {}
        authors = base_meta.get("authors")
        if not isinstance(authors, list):
            authors = info.get("authors")
        author_text = base_meta.get("author")
        if not author_text and isinstance(authors, list) and authors:
            author_text = "、".join([str(name).strip() for name in authors if str(name).strip()])

        normalized = self._normalize_textbook_meta(
            {
                "textbook_role": base_meta.get("textbook_role")
                or info.get("textbook_role")
                or info.get("role"),
                "subject": base_meta.get("subject") or subject or info.get("subject"),
                "title": base_meta.get("title")
                or title
                or info.get("book_title")
                or metadata.get("source_file"),
                "edition": base_meta.get("edition") or info.get("edition"),
                "academic_year": base_meta.get("academic_year") or info.get("academic_year"),
                "author": author_text,
                "authors": authors,
                "source_file": base_meta.get("source_file") or source_file or info.get("source_file"),
                "priority_score": base_meta.get("priority_score", info.get("priority_score", 0.0)),
            }
        )
        if not normalized.get("textbook_role"):
            normalized["textbook_role"] = self._infer_textbook_role_from_text(
                f"{normalized.get('title') or ''} {normalized.get('source_file') or ''} {normalized.get('edition') or ''}"
            )
        normalized["priority_score"] = self._compute_textbook_priority_score(normalized)
        return normalized

    def _normalize_syllabus_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {field: None for field in self.SYLLABUS_PRIORITY_FIELDS}
        normalized["is_primary"] = self._to_bool(meta.get("is_primary"), False)
        normalized["priority_score"] = self._to_float(meta.get("priority_score"), 0.0)
        for field in self.SYLLABUS_PRIORITY_FIELDS:
            if field in {"is_primary", "priority_score"}:
                continue
            value = str(meta.get(field) or "").strip()
            normalized[field] = value or None
        return normalized

    def _normalize_textbook_meta(self, meta: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(meta, dict):
            meta = {}
        normalized: Dict[str, Any] = {field: None for field in self.TEXTBOOK_PRIORITY_FIELDS}
        role = str(meta.get("textbook_role") or "").strip().lower()
        normalized["textbook_role"] = role if role in {"main", "supplementary"} else None
        normalized["priority_score"] = self._to_float(meta.get("priority_score"), 0.0)
        normalized["authors"] = []

        authors = meta.get("authors")
        if isinstance(authors, list):
            normalized["authors"] = self._deduplicate(authors)[:10]
        author = str(meta.get("author") or "").strip()
        if author:
            normalized["author"] = author
        elif normalized["authors"]:
            normalized["author"] = "、".join(normalized["authors"])

        for field in ["subject", "title", "edition", "academic_year", "source_file"]:
            value = str(meta.get(field) or "").strip()
            normalized[field] = value or None
        return normalized

    def _compute_syllabus_priority_score(self, meta: Dict[str, Any]) -> float:
        score = self._to_float(meta.get("priority_score"), 0.0)
        if self._to_bool(meta.get("is_primary"), False):
            score += 100.0
        if meta.get("course_code"):
            score += 12.0
        if meta.get("course_name"):
            score += 8.0
        if meta.get("school"):
            score += 4.0
        if meta.get("department"):
            score += 3.0
        if meta.get("major"):
            score += 5.0
        score += self._normalize_date_or_year_score(meta.get("academic_year"), scale=0.2, cap=5.0)
        score += self._normalize_date_or_year_score(meta.get("effective_date"), scale=0.05, cap=4.0)
        score += self._normalize_version_score(meta.get("version"))
        return round(score, 4)

    def _compute_textbook_priority_score(self, meta: Dict[str, Any]) -> float:
        score = self._to_float(meta.get("manual_priority"), 0.0)
        role = str(meta.get("textbook_role") or "").strip().lower()
        if role == "main":
            score += 100.0
        elif role == "supplementary":
            score += 45.0
        else:
            score += 60.0
        if meta.get("subject"):
            score += 8.0
        if meta.get("title"):
            score += 6.0
        if meta.get("author") or meta.get("authors"):
            score += 3.0
        score += self._normalize_date_or_year_score(meta.get("academic_year"), scale=0.2, cap=6.0)
        score += self._normalize_version_score(meta.get("edition"))
        return round(score, 4)

    def _infer_textbook_role_from_text(self, text: str) -> str:
        value = str(text or "").strip().lower()
        if not value:
            return "main"
        if "目录" in value or "contents" in value:
            return "supplementary"
        if any(token in value for token in ["习题", "练习", "辅导", "参考", "题解", "workbook", "supplementary"]):
            return "supplementary"
        return "main"

    def _is_directory_like_title(self, value: Any) -> bool:
        title = str(value or "").strip().lower()
        compact = re.sub(r"\s+", "", title)
        if compact in {"目录", "目 录", "contents", "content"}:
            return True
        if "目录" in compact or "contents" in compact:
            return True
        return False

    def _normalize_date_or_year_score(self, value: Optional[str], scale: float, cap: float) -> float:
        text = str(value or "").strip()
        if not text:
            return 0.0
        years = re.findall(r"(20\d{2}|19\d{2})", text)
        if not years:
            return 0.0
        latest = max(int(item) for item in years)
        return min(latest * scale, cap)

    def _normalize_version_score(self, version: Optional[str]) -> float:
        text = str(version or "").strip().lower()
        if not text:
            return 0.0
        nums = re.findall(r"\d+(?:\.\d+)?", text)
        if nums:
            return min(float(nums[-1]) * 0.5, 3.0)
        if "new" in text or "latest" in text or "新版" in text:
            return 2.0
        if "old" in text or "旧版" in text:
            return -1.0
        return 0.0

    def _to_bool(self, value: Any, default: bool) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

    def _to_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except Exception:
            return default

    def _first_list_item(self, value: Any) -> Optional[str]:
        if not isinstance(value, list):
            return None
        for item in value:
            text = str(item or "").strip()
            if text:
                return text
        return None

    def _deduplicate(self, values: List[Any]) -> List[str]:
        result: List[str] = []
        seen = set()
        for value in values:
            text = str(value).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            result.append(text)
        return result

    @staticmethod
    def extract_domain(value: Optional[str]) -> Optional[str]:
        """提取 URL 域名（用于热点静态源策略）。"""
        if not value:
            return None
        if not (value.startswith("http://") or value.startswith("https://")):
            return None
        return (urlparse(value).netloc or "").lower() or None

