#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> mn, lazy;
    vector<long long> msum;

    SegTree(const vector<long long>& w) {
        n = (int)w.size() - 1;
        mn.assign(4 * n + 4, 0);
        lazy.assign(4 * n + 4, 0);
        msum.assign(4 * n + 4, 0);
        build(1, 1, n, w);
    }
    void pull(int u) {
        int a = mn[2 * u], b = mn[2 * u + 1];
        mn[u] = min(a, b);
        msum[u] = (a == mn[u] ? msum[2 * u] : 0) + (b == mn[u] ? msum[2 * u + 1] : 0);
    }
    void build(int u, int l, int r, const vector<long long>& w) {
        if (l == r) {
            mn[u] = 0;
            msum[u] = w[l];
            return;
        }
        int m = (l + r) >> 1;
        build(2 * u, l, m, w);
        build(2 * u + 1, m + 1, r, w);
        pull(u);
    }
    void apply(int u, int d) {
        mn[u] += d;
        lazy[u] += d;
    }
    void push(int u) {
        if (lazy[u] != 0) {
            apply(2 * u, lazy[u]);
            apply(2 * u + 1, lazy[u]);
            lazy[u] = 0;
        }
    }
    void add(int u, int l, int r, int ql, int qr, int d) {
        if (ql <= l && r <= qr) {
            apply(u, d);
            return;
        }
        push(u);
        int m = (l + r) >> 1;
        if (ql <= m) add(2 * u, l, m, ql, qr, d);
        if (qr > m) add(2 * u + 1, m + 1, r, ql, qr, d);
        pull(u);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> w(n + 1);
    long long total = 0;
    for (int i = 1; i <= n; i++) {
        cin >> w[i];
        total += w[i];
    }
    SegTree st(w);
    while (q--) {
        int t, l, r;
        cin >> t >> l >> r;
        st.add(1, 1, n, l, r, t == 1 ? +1 : -1);
        long long zero_len = (st.mn[1] == 0 ? st.msum[1] : 0LL);
        cout << (total - zero_len) << '\n';
    }
    return 0;
}
