from app.routing.domain_router import TreeDomainRouter, build_default_company_tree


def test_router_prefers_smartph_branch() -> None:
    router = TreeDomainRouter(build_default_company_tree())

    routed = router.route("Thông số của SmartpH-06M trong hệ thống quan trắc nước thải?")

    assert "Thiết Bị Quan Trắc Môi Trường" in routed.branch_path
    assert "Hệ Thống Quan Trắc Nước Thải Tự Động" in routed.branch_path
    assert routed.confidence > 0


def test_router_falls_back_to_root_when_unclear() -> None:
    router = TreeDomainRouter(build_default_company_tree())

    routed = router.route("Một câu hỏi không liên quan")

    assert routed.branch_path == ["Công ty Cổ Phần KHKT Phượng Hải"]
    assert routed.confidence == 0

