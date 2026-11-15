#include <bits/stdc++.h>
using namespace std;

static constexpr long long NEG_INF = -(1LL << 60);

struct SegTree {
    int n;
    vector<long long> mx, lz;
    SegTree(int n) : n(n), mx(4 * n + 5, NEG_INF), lz(4 * n + 5, 0) {}

    void apply(int u, long long v) {
        mx[u] += v;
        lz[u] += v;
    }
    void push(int u) {
        if (lz[u] != 0) {
            apply(u << 1, lz[u]);
            apply(u << 1 | 1, lz[u]);
            lz[u] = 0;
        }
    }
    void set_point(int u, int l, int r, int pos, long long val) {
        if (l == r) { mx[u] = val; lz[u] = 0; return; }
        push(u);
        int mid = (l + r) >> 1;
        if (pos <= mid) set_point(u << 1, l, mid, pos, val);
        else set_point(u << 1 | 1, mid + 1, r, pos, val);
        mx[u] = max(mx[u << 1], mx[u << 1 | 1]);
    }
    void add_range(int u, int l, int r, int ql, int qr, long long val) {
        if (ql <= l && r <= qr) { apply(u, val); return; }
        push(u);
        int mid = (l + r) >> 1;
        if (ql <= mid) add_range(u << 1, l, mid, ql, qr, val);
        if (qr > mid) add_range(u << 1 | 1, mid + 1, r, ql, qr, val);
        mx[u] = max(mx[u << 1], mx[u << 1 | 1]);
    }
    long long query_max(int u, int l, int r, int ql, int qr) {
        if (ql <= l && r <= qr) return mx[u];
        push(u);
        int mid = (l + r) >> 1;
        long long res = NEG_INF;
        if (ql <= mid) res = max(res, query_max(u << 1, l, mid, ql, qr));
        if (qr > mid) res = max(res, query_max(u << 1 | 1, mid + 1, r, ql, qr));
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> c(n + 1);
    for (int i = 1; i <= n; i++) cin >> c[i];
    vector<vector<pair<int, long long>>> by_r(n + 1);
    for (int i = 0; i < m; i++) {
        int l, r; long long p;
        cin >> l >> r >> p;
        by_r[r].push_back({l, p});
    }

    SegTree st(n);
    vector<long long> f(n + 1, 0);
    for (int r = 1; r <= n; r++) {
        long long base = (r >= 2 ? f[r - 2] : 0LL);
        st.set_point(1, 1, n, r, base);
        st.add_range(1, 1, n, 1, r, -c[r]);
        for (auto [l, p] : by_r[r]) {
            st.add_range(1, 1, n, 1, l, p);
        }
        f[r] = max(f[r - 1], st.query_max(1, 1, n, 1, r));
    }
    cout << f[n] << '\n';
    return 0;
}
