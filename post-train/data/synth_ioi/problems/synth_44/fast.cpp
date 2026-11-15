#include <bits/stdc++.h>
using namespace std;

struct SegBeats {
    int n;
    vector<long long> sm, mx;
    SegBeats(const vector<long long>& a) {
        n = (int)a.size() - 1;
        sm.assign(4 * n + 4, 0);
        mx.assign(4 * n + 4, 0);
        build(1, 1, n, a);
    }
    void pull(int u) {
        sm[u] = sm[2 * u] + sm[2 * u + 1];
        mx[u] = max(mx[2 * u], mx[2 * u + 1]);
    }
    void build(int u, int l, int r, const vector<long long>& a) {
        if (l == r) { sm[u] = mx[u] = a[l]; return; }
        int m = (l + r) >> 1;
        build(2 * u, l, m, a);
        build(2 * u + 1, m + 1, r, a);
        pull(u);
    }
    void range_mod(int u, int l, int r, int ql, int qr, long long mod) {
        if (r < ql || qr < l || mx[u] < mod) return;
        if (l == r) { sm[u] %= mod; mx[u] = sm[u]; return; }
        int m = (l + r) >> 1;
        range_mod(2 * u, l, m, ql, qr, mod);
        range_mod(2 * u + 1, m + 1, r, ql, qr, mod);
        pull(u);
    }
    void point_set(int u, int l, int r, int p, long long v) {
        if (l == r) { sm[u] = mx[u] = v; return; }
        int m = (l + r) >> 1;
        if (p <= m) point_set(2 * u, l, m, p, v);
        else point_set(2 * u + 1, m + 1, r, p, v);
        pull(u);
    }
    long long query_sum(int u, int l, int r, int ql, int qr) const {
        if (r < ql || qr < l) return 0;
        if (ql <= l && r <= qr) return sm[u];
        int m = (l + r) >> 1;
        return query_sum(2 * u, l, m, ql, qr) + query_sum(2 * u + 1, m + 1, r, ql, qr);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    SegBeats st(a);
    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int l, r; long long m;
            cin >> l >> r >> m;
            st.range_mod(1, 1, n, l, r, m);
        } else if (t == 2) {
            int p; long long x;
            cin >> p >> x;
            st.point_set(1, 1, n, p, x);
        } else {
            int l, r;
            cin >> l >> r;
            cout << st.query_sum(1, 1, n, l, r) << '\n';
        }
    }
    return 0;
}
