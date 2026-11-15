#include <bits/stdc++.h>
using namespace std;

static const int MOD = 998244353;

struct Val {
    long long v;
    int ways;
};

static inline Val merge_val(const Val& a, const Val& b) {
    if (a.v > b.v) return a;
    if (b.v > a.v) return b;
    int w = a.ways + b.ways;
    if (w >= MOD) w -= MOD;
    return {a.v, w};
}

struct Pt {
    int x, y, z;
    long long w;
    int id;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<Pt> pts(n);
    vector<int> zs(n);
    for (int i = 0; i < n; i++) {
        cin >> pts[i].x >> pts[i].y >> pts[i].z >> pts[i].w;
        pts[i].id = i;
        zs[i] = pts[i].z;
    }
    sort(zs.begin(), zs.end());
    zs.erase(unique(zs.begin(), zs.end()), zs.end());
    int mz = (int)zs.size();
    for (int i = 0; i < n; i++) {
        pts[i].z = (int)(lower_bound(zs.begin(), zs.end(), pts[i].z) - zs.begin()) + 1;
    }

    sort(pts.begin(), pts.end(), [](const Pt& a, const Pt& b) {
        if (a.x != b.x) return a.x < b.x;
        if (a.y != b.y) return a.y < b.y;
        return a.z < b.z;
    });

    // Group strictly by distinct X so left groups always have X_left < X_right
    vector<pair<int, int>> groups;
    for (int i = 0; i < n;) {
        int j = i;
        while (j < n && pts[j].x == pts[i].x) j++;
        groups.push_back({i, j});
        i = j;
    }

    vector<Val> dp(n, {0, 0});
    for (int i = 0; i < n; i++) {
        dp[pts[i].id] = {pts[i].w, 1};
    }

    vector<Val> bit(mz + 2, {0, 0});
    auto bit_upd = [&](int idx, const Val& val) {
        for (; idx <= mz; idx += idx & -idx) bit[idx] = merge_val(bit[idx], val);
    };
    auto bit_qry = [&](int idx) -> Val {
        Val res{0, 0};
        for (; idx > 0; idx -= idx & -idx) res = merge_val(res, bit[idx]);
        return res;
    };
    auto bit_clr = [&](int idx) {
        for (; idx <= mz; idx += idx & -idx) {
            if (bit[idx].v == 0 && bit[idx].ways == 0) break;
            bit[idx] = {0, 0};
        }
    };

    vector<Pt> left_buf, right_buf;
    left_buf.reserve(n);
    right_buf.reserve(n);

    auto cdq = [&](auto& self, int gl, int gr) -> void {
        if (gl >= gr) return;
        int gm = (gl + gr) >> 1;
        self(self, gl, gm);

        int l_start = groups[gl].first, l_end = groups[gm].second;
        int r_start = groups[gm + 1].first, r_end = groups[gr].second;
        left_buf.assign(pts.begin() + l_start, pts.begin() + l_end);
        right_buf.assign(pts.begin() + r_start, pts.begin() + r_end);

        auto cmp_y = [](const Pt& a, const Pt& b) {
            if (a.y != b.y) return a.y < b.y;
            return a.z < b.z;
        };
        sort(left_buf.begin(), left_buf.end(), cmp_y);
        sort(right_buf.begin(), right_buf.end(), cmp_y);

        int p = 0, lsz = (int)left_buf.size();
        for (const auto& rq : right_buf) {
            while (p < lsz && left_buf[p].y < rq.y) {
                bit_upd(left_buf[p].z, dp[left_buf[p].id]);
                p++;
            }
            Val best = bit_qry(rq.z - 1);
            if (best.ways > 0) {
                Val cand{best.v + rq.w, best.ways};
                dp[rq.id] = merge_val(dp[rq.id], cand);
            }
        }
        for (int i = 0; i < p; i++) bit_clr(left_buf[i].z);

        self(self, gm + 1, gr);
    };

    if (!groups.empty()) cdq(cdq, 0, (int)groups.size() - 1);

    Val ans{0, 0};
    for (int i = 0; i < n; i++) ans = merge_val(ans, dp[i]);
    cout << ans.v << ' ' << ans.ways << '\n';
    return 0;
}
