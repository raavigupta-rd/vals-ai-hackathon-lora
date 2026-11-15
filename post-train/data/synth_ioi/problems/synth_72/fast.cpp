#include <bits/stdc++.h>
using namespace std;

struct SegTree {
    int n;
    vector<int> mn, cnt, lz;
    SegTree(int n) : n(n), mn(4 * n + 5, 1e9), cnt(4 * n + 5, 0), lz(4 * n + 5, 0) {}

    void apply(int u, int v) {
        mn[u] += v;
        lz[u] += v;
    }
    void push(int u) {
        if (lz[u] != 0) {
            apply(u << 1, lz[u]);
            apply(u << 1 | 1, lz[u]);
            lz[u] = 0;
        }
    }
    void pull(int u) {
        int a = u << 1, b = u << 1 | 1;
        mn[u] = min(mn[a], mn[b]);
        cnt[u] = (mn[a] == mn[u] ? cnt[a] : 0) + (mn[b] == mn[u] ? cnt[b] : 0);
    }
    void activate(int u, int l, int r, int pos, int val) {
        if (l == r) {
            mn[u] = val;
            cnt[u] = 1;
            lz[u] = 0;
            return;
        }
        push(u);
        int mid = (l + r) >> 1;
        if (pos <= mid) activate(u << 1, l, mid, pos, val);
        else activate(u << 1 | 1, mid + 1, r, pos, val);
        pull(u);
    }
    void add(int u, int l, int r, int ql, int qr, int val) {
        if (ql <= l && r <= qr) {
            apply(u, val);
            return;
        }
        push(u);
        int mid = (l + r) >> 1;
        if (ql <= mid) add(u << 1, l, mid, ql, qr, val);
        if (qr > mid) add(u << 1 | 1, mid + 1, r, ql, qr, val);
        pull(u);
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> p(n + 1);
    for (int i = 1; i <= n; i++) cin >> p[i];

    SegTree st(n);
    // Stack entries: (pos, val)
    vector<pair<int, int>> smx = {{0, INT_MAX}}, smn = {{0, INT_MIN}};
    long long ans = 0;

    for (int r = 1; r <= n; r++) {
        while (smx.back().second < p[r]) {
            int rpos = smx.back().first, old_v = smx.back().second;
            smx.pop_back();
            int lpos = smx.back().first + 1;
            st.add(1, 1, n, lpos, rpos, p[r] - old_v);
        }
        smx.push_back({r, p[r]});

        while (smn.back().second > p[r]) {
            int rpos = smn.back().first, old_v = smn.back().second;
            smn.pop_back();
            int lpos = smn.back().first + 1;
            st.add(1, 1, n, lpos, rpos, old_v - p[r]);
        }
        smn.push_back({r, p[r]});

        st.activate(1, 1, n, r, r);
        if (st.mn[1] == r) ans += st.cnt[1];
    }

    cout << ans << '\n';
    return 0;
}
