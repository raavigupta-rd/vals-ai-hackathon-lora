#include <bits/stdc++.h>
using namespace std;

static constexpr long long INF = (1LL << 61);

struct SegTree {
    int L0, R0;
    vector<long long> mn, lz;
    SegTree(int l0, int r0) : L0(l0), R0(r0) {
        int sz = max(1, r0 - l0 + 1);
        mn.assign(4 * sz + 5, INF);
        lz.assign(4 * sz + 5, 0);
    }
    void apply(int u, long long v) { mn[u] += v; lz[u] += v; }
    void push(int u) {
        if (lz[u] != 0) {
            apply(u << 1, lz[u]);
            apply(u << 1 | 1, lz[u]);
            lz[u] = 0;
        }
    }
    void build(int u, int l, int r, const vector<long long>& init) {
        if (l == r) { mn[u] = init[l]; return; }
        int mid = (l + r) >> 1;
        build(u << 1, l, mid, init);
        build(u << 1 | 1, mid + 1, r, init);
        mn[u] = min(mn[u << 1], mn[u << 1 | 1]);
    }
    void add(int u, int l, int r, int ql, int qr, long long val) {
        if (ql > qr) return;
        if (ql <= l && r <= qr) { apply(u, val); return; }
        push(u);
        int mid = (l + r) >> 1;
        if (ql <= mid) add(u << 1, l, mid, ql, qr, val);
        if (qr > mid) add(u << 1 | 1, mid + 1, r, ql, qr, val);
        mn[u] = min(mn[u << 1], mn[u << 1 | 1]);
    }
    long long query(int u, int l, int r, int ql, int qr) {
        if (ql > qr) return INF;
        if (ql <= l && r <= qr) return mn[u];
        push(u);
        int mid = (l + r) >> 1;
        long long res = INF;
        if (ql <= mid) res = min(res, query(u << 1, l, mid, ql, qr));
        if (qr > mid) res = min(res, query(u << 1 | 1, mid + 1, r, ql, qr));
        return res;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, m;
    cin >> n >> m;
    vector<long long> cap(m + 2, 0), pref(m + 2, 0), suff(m + 3, 0);
    for (int i = 1; i <= m; i++) {
        cin >> cap[i];
        pref[i] = pref[i - 1] + cap[i];
    }
    for (int i = m; i >= 1; i--) suff[i] = suff[i + 1] + cap[i];

    vector<vector<pair<int, long long>>> by_l(m + 1);
    long long tot_c = 0;
    for (int i = 0; i < n; i++) {
        int l, r; long long c;
        cin >> l >> r >> c;
        tot_c += c;
        if (l < m) by_l[l].push_back({r, c});
    }

    // Cut Case 1: all piers cut (capacity pref[m]), or all fleets cut (tot_c)
    long long ans = min(tot_c, pref[m]);

    // Cut Case 2: uncut piers are [a+1 .. b-1] for 0 <= a <= m-1 and a+2 <= b <= m+1
    SegTree st(2, m + 1);
    st.build(1, 2, m + 1, suff);

    for (int a = 0; a <= m - 1; a++) {
        for (auto [r, c] : by_l[a]) {
            if (r >= 2) st.add(1, 2, m + 1, 2, min(r, m + 1), -c);
        }
        long long best_b = st.query(1, 2, m + 1, a + 2, m + 1);
        ans = min(ans, pref[a] + tot_c + best_b);
    }

    cout << ans << '\n';
    return 0;
}
