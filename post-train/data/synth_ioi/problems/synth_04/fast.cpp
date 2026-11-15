#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    struct Pt { int x, y; long long w; };
    vector<Pt> pts(n);
    vector<int> ys;
    ys.reserve(n);
    for (int i = 0; i < n; i++) {
        cin >> pts[i].x >> pts[i].y >> pts[i].w;
        ys.push_back(pts[i].y);
    }
    sort(ys.begin(), ys.end());
    ys.erase(unique(ys.begin(), ys.end()), ys.end());
    int m = (int)ys.size();

    struct Ev { int x, y1, y2, id, sign; };
    vector<Ev> evs;
    evs.reserve(2 * q);
    for (int i = 0; i < q; i++) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        int ly = (int)(lower_bound(ys.begin(), ys.end(), y1) - ys.begin());
        int ry = (int)(upper_bound(ys.begin(), ys.end(), y2) - ys.begin());
        if (ly < ry) {
            evs.push_back({x2, ly, ry, i, +1});
            evs.push_back({x1 - 1, ly, ry, i, -1});
        }
    }
    sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) { return a.x < b.x; });
    sort(evs.begin(), evs.end(), [](const Ev& a, const Ev& b) { return a.x < b.x; });

    vector<long long> bit(m + 1, 0);
    auto add = [&](int idx, long long v) {
        for (++idx; idx <= m; idx += idx & -idx) bit[idx] += v;
    };
    auto sum = [&](int idx) {
        long long s = 0;
        for (; idx > 0; idx -= idx & -idx) s += bit[idx];
        return s;
    };

    vector<long long> ans(q, 0);
    int p = 0;
    for (const auto& e : evs) {
        while (p < n && pts[p].x <= e.x) {
            int yi = (int)(lower_bound(ys.begin(), ys.end(), pts[p].y) - ys.begin());
            add(yi, pts[p].w);
            p++;
        }
        ans[e.id] += e.sign * (sum(e.y2) - sum(e.y1));
    }
    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
