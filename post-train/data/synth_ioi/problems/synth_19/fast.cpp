#include <bits/stdc++.h>
using namespace std;

const long long INF = (1LL << 61);

struct SegTree {
    int sz;
    vector<long long> t;
    SegTree(int n) {
        sz = 1;
        while (sz <= n + 1) sz <<= 1;
        t.assign(2 * sz, INF);
    }
    void chmin(int idx, long long val) {
        int u = sz + idx;
        if (val >= t[u]) return;
        t[u] = val;
        for (u >>= 1; u >= 1; u >>= 1) t[u] = min(t[2 * u], t[2 * u + 1]);
    }
    long long query(int l, int r) const { // inclusive [l..r]
        long long res = INF;
        for (l += sz, r += sz + 1; l < r; l >>= 1, r >>= 1) {
            if (l & 1) res = min(res, t[l++]);
            if (r & 1) res = min(res, t[--r]);
        }
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    struct Seg { int l, r; long long w; };
    vector<Seg> a(m);
    for (int i = 0; i < m; i++) cin >> a[i].l >> a[i].r >> a[i].w;
    sort(a.begin(), a.end(), [](const Seg& x, const Seg& y) { return x.r < y.r; });

    SegTree st(n);
    st.chmin(0, 0);
    for (const auto& s : a) {
        long long best = st.query(s.l - 1, s.r);
        if (best < INF) st.chmin(s.r, best + s.w);
    }
    long long ans = st.query(n, n);
    cout << (ans >= INF ? -1LL : ans) << '\n';
    return 0;
}
