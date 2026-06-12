from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from app.schemas.retrieval import RoutedQuery


@dataclass(frozen=True)
class DomainNode:
    name: str
    keywords: tuple[str, ...] = ()
    children: tuple["DomainNode", ...] = field(default_factory=tuple)


class DomainRouter(Protocol):
    def route(self, question: str) -> RoutedQuery:
        """Route a question to the most relevant branch before retrieval."""
        ...


class TreeDomainRouter:
    def __init__(self, root: DomainNode, min_confidence: float = 0.15) -> None:
        self.root = root
        self.min_confidence = min_confidence

    def route(self, question: str) -> RoutedQuery:
        normalized = question.lower()
        path, score = self._best_path(self.root, normalized)
        confidence = min(score / 6.0, 1.0)

        if confidence < self.min_confidence:
            path = [self.root.name]
            confidence = 0.0
            rationale = "No strong branch match; use root-level constrained retrieval."
        else:
            rationale = "Matched branch keywords in the hierarchy."

        return RoutedQuery(
            query=question,
            branch_path=path,
            confidence=confidence,
            rationale=rationale,
        )

    def _best_path(self, node: DomainNode, question: str) -> tuple[list[str], float]:
        own_score = self._score_node(node, question)
        best_path = [node.name]
        best_score = own_score

        for child in node.children:
            child_path, child_score = self._best_path(child, question)
            combined_score = own_score + child_score
            if combined_score > best_score:
                best_score = combined_score
                best_path = [node.name, *child_path]

        return best_path, best_score

    @staticmethod
    def _score_node(node: DomainNode, question: str) -> float:
        return sum(1.0 for keyword in node.keywords if keyword.lower() in question)


def build_default_company_tree() -> DomainNode:
    return DomainNode(
        name="Công ty Cổ Phần KHKT Phượng Hải",
        keywords=("phượng hải", "company", "công ty", "iso"),
        children=(
            DomainNode(
                name="Thông Tin Chung",
                keywords=("giới thiệu", "liên hệ", "địa chỉ", "email", "website", "iso"),
            ),
            DomainNode(
                name="Thiết Bị Quan Trắc Môi Trường",
                keywords=("smartph", "quan trắc", "nước thải", "transmitter", "controller"),
                children=(
                    DomainNode(
                        name="Thiết Bị Quan Trắc Nước Đơn Lẻ",
                        keywords=("smartph-01", "ammonia", "nh4", "ph/orp", "cod"),
                    ),
                    DomainNode(
                        name="Hệ Thống Quan Trắc Nước Thải Tự Động",
                        keywords=("smartph-06m", "smartph-log", "smartph-ws1", "lấy mẫu"),
                    ),
                ),
            ),
            DomainNode(
                name="Nội Thất & Thiết Bị Phòng Thí Nghiệm",
                keywords=("bestlab", "phòng thí nghiệm", "tủ hút", "bàn thí nghiệm"),
                children=(
                    DomainNode(
                        name="Tủ Hút",
                        keywords=("tủ hút", "greenlab", "phlab", "acid", "hóa chất"),
                    ),
                    DomainNode(
                        name="Tủ Đựng Hóa Chất",
                        keywords=("tủ đựng hóa chất", "chống cháy", "khử mùi", "acid"),
                    ),
                    DomainNode(
                        name="Bàn Thí Nghiệm",
                        keywords=("bàn thí nghiệm", "phenolic", "hpl", "mặt bàn"),
                    ),
                ),
            ),
            DomainNode(
                name="TÌNH HUỐNG - HƯỚNG XỬ LÝ CÁC PHÒNG BAN",
                keywords=("tình huống", "xử lý", "phòng ban", "quy trình", "sop"),
                children=(
                    DomainNode(
                        name="TÌNH HUỐNG PHÒNG KINH DOANH BESTLAB",
                        keywords=("kinh doanh bestlab", "báo giá bestlab", "khách hàng bestlab"),
                    ),
                    DomainNode(
                        name="TÌNH HUỐNG PHÒNG KINH DOANH SMARTPH",
                        keywords=("kinh doanh smartph", "báo giá smartph", "khách hàng smartph"),
                    ),
                    DomainNode(
                        name="TÌNH HUỐNG PHÒNG MUA HÀNG",
                        keywords=("mua hàng", "nhà cung cấp", "đặt hàng", "采购"),
                    ),
                ),
            ),
        ),
    )
