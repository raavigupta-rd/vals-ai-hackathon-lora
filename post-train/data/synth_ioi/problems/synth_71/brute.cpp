#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    struct Rect { int x1, y1, x2, y2; };
    vector<Rect> rs(n);
    vector<int> xs;
    xs.reserve(2 * n);
    for (int i = 0; i < n; i++) {
        cin >> rs[i].x1 >> rs[i].y1 >> rs[i].x2 >> rs[i].y2;
        xs.push_back(rs[i].x1);
        xs.push_back(rs[i].x2);
    }
    sort(xs.begin(), xs.end());
    xs.erase(unique(xs.begin(), xs.end()), xs.end());

    long long ans = 0;
    vector<pair<int, int>> segs;
    segs.reserve(n);
    for (size_t i = 0; i + 1 < xs.size(); i++) {
        int xl = xs[i], xr = xs[i + 1];
        segs.clear();
        for (const auto& rc : rs) {
            if (rc.x1 <= xl && xr <= rc.x2) segs.push_back({rc.y1, rc.y2});
        }
        if (segs.empty()) continue;
        sort(segs.begin(), segs.end());
        long long ylen = 0;
        int cl = segs[0].first, cr = segs[0].second;
        for (size_t j = 1; j < segs.size(); j++) {
            if (segs[j].first <= cr) cr = max(cr, segs[j].second);
            else { ylen += cr - cl; cl = segs[j].first; cr = segs[j].second; }
        }
        ylen += cr - cl;
        ans += ylen * (xr - xl);
    }
    cout << ans << '\n';
    return 0;
}
