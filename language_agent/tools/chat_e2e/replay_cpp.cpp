#include <charconv>
#include <clocale>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "colour_name.h"
static double num(const std::string& s){ double v=0; std::from_chars(s.data(), s.data()+s.size(), v); return v; }
int main(int argc, char** argv)
{
    std::setlocale(LC_ALL, "");
    for (int f = 1; f < argc; ++f)
    {
        std::ifstream in(argv[f]); std::string line; std::getline(in, line);
        std::map<std::string, rc::appearance::AppearanceBelief> bel; std::map<std::string, double> last_t;
        std::map<std::string, std::map<std::string,int>> names; std::map<std::string,float> minp; std::map<std::string,int> down;
        while (std::getline(in, line))
        {
            std::vector<std::string> c; std::stringstream ss(line); std::string x; while (std::getline(ss, x, ',')) c.push_back(x);
            if (c.size() < 12) continue;
            const std::string node = c[1]; const double t = num(c[0]) / 1000.0;
            auto& a = bel[node];
            if (last_t.contains(node)) a.inflate_for_age(float(t - last_t[node]));
            last_t[node] = t;
            a.observe({float(num(c[2])), float(num(c[3])), float(num(c[4]))}, {float(num(c[5])), float(num(c[6])), float(num(c[7]))}, float(num(c[8])),
                      float(num(c[9])), float(num(c[10])), float(num(c[11])));
            const auto cn = rc::appearance::name_colour(a);
            names[node][cn.name]++; if (not minp.contains(node) or cn.prob < minp[node]) minp[node] = cn.prob;
            if (a.last_weight() < 0.5f) down[node]++;
        }
        std::printf("== %s\n", argv[f]);
        for (auto& [n, m] : names) { std::printf("  %s min p %.2f, frames weighed down %d:", n.c_str(), minp[n], down[n]); for (auto& [k, v] : m) std::printf(" %s %d", k.c_str(), v); std::printf("\n"); }
    }
}
