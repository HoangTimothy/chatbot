from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentBranch:
    path: list[str]
    aliases: list[str] = field(default_factory=list)


class HierarchyResolver:
    def __init__(self, branches: list[DocumentBranch]) -> None:
        self.branches = branches

    def resolve(self, document_title: str) -> list[str]:
        normalized = document_title.lower()
        for branch in self.branches:
            candidates = [*branch.path, *branch.aliases]
            if any(candidate.lower() in normalized for candidate in candidates):
                return branch.path
        return []

