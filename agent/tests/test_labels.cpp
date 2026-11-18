#include <catch2/catch_test_macros.hpp>
#include <regex>

#include "inference.hpp"
#include "support.hpp"

using namespace bunnycam;

// kLabels is a copy of server/shared/labels.py LABELS. Fail on drift.
TEST_CASE("kLabels matches shared/labels.py") {
    const std::string py = test::read_file(test::kRepoDir / "server" / "shared" / "labels.py");
    std::smatch block;
    REQUIRE(std::regex_search(py, block, std::regex(R"(\nLABELS = \[([^\]]*)\])")));
    std::vector<std::string> labels;
    const std::string body = block[1];
    const std::regex quoted(R"x("([^"]+)")x");
    for (auto it = std::sregex_iterator(body.begin(), body.end(), quoted);
         it != std::sregex_iterator(); ++it) {
        labels.push_back((*it)[1]);
    }
    CHECK(labels == kLabels);

    std::smatch resting;
    REQUIRE(std::regex_search(py, resting, std::regex(R"(\nRESTING_LABELS = [^\n]*\n?[^\n]*)")));
    for (const auto& label : kRestingLabels) {
        CHECK(resting.str().find('"' + label + '"') != std::string::npos);
    }
}
