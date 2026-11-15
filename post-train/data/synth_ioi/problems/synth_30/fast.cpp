#include <bits/stdc++.h>
using namespace std;

struct RangeBIT {
    int n;
    vector<long long> t;
    RangeBIT(int n) : n(n), t(n + 2, 0) {}
    void reset() { fill(t.begin(), t.end(), 0); }
    void add_pt(int i, long long v) {
        for (; i <= n + 1; i += i & -i) t[i] += v;
    }
    void add_range(int l, int r, long long v) {
        add_pt(l, v);
        add_pt(r + 1, -v);
    }
    long long get(int i) const {
        long long s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m, q;
    cin >> n >> m >> q;
    vector<vector<int>> yards(n + 1);
    for (int j = 1; j <= m; j++) {
        int o; cin >> o;
        yards[o].push_back(j);
    }
    vector<long long> req(n + 1);
    for (int i = 1; i <= n; i++) cin >> req[i];
    struct Upd { int l, r; long long w; };
    vector<Upd> upds(q + 1);
    for (int t = 1; t <= q; t++) cin >> upds[t].l >> upds[t].r >> upds[t].w;

    vector<int> lo(n + 1, 1), hi(n + 1, q + 1);
    RangeBIT bit(m);
    vector<vector<int>> buckets(q + 2);

    while (true) {
        bool any = false;
        for (int t = 1; t <= q; t++) buckets[t].clear();
        for (int i = 1; i <= n; i++) {
            if (lo[i] < hi[i]) {
                any = true;
                buckets[(lo[i] + hi[i]) >> 1].push_back(i);
            }
        }
        if (!any) break;
        bit.reset();
        for (int t = 1; t <= q; t++) {
            bit.add_range(upds[t].l, upds[t].r, upds[t].w);
            for (int prov : buckets[t]) {
                __int128 sum = 0;
                for (int y : yards[prov]) {
                    sum += bit.get(y);
                    if (sum >= req[prov]) break;
                }
                if (sum >= req[prov]) hi[prov] = t;
                else lo[prov] = t + 1;
            }
        }
    }
    for (int i = 1; i <= n; i++) {
        cout << (lo[i] <= q ? lo[i] : -1) << '\n';
    }
    return 0;
}
