#include <bits/stdc++.h>
using namespace std;

const long long NEG = -(1LL << 60);
struct Node {
    long long sum, pref, suff, best;
};

static inline Node make_leaf(long long v) { return {v, v, v, v}; }
static inline Node merge_node(const Node& a, const Node& b) {
    return {
        a.sum + b.sum,
        max(a.pref, a.sum + b.pref),
        max(b.suff, b.sum + a.suff),
        max({a.best, b.best, a.suff + b.pref})
    };
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    int sz = 1;
    while (sz < n) sz <<= 1;
    vector<Node> seg(2 * sz, {0, NEG, NEG, NEG});
    for (int i = 0; i < n; i++) {
        long long x; cin >> x;
        seg[sz + i] = make_leaf(x);
    }
    for (int i = sz - 1; i >= 1; i--) seg[i] = merge_node(seg[2 * i], seg[2 * i + 1]);

    while (q--) {
        int t; cin >> t;
        if (t == 1) {
            int p; long long x;
            cin >> p >> x;
            int u = sz + p - 1;
            seg[u] = make_leaf(x);
            for (u >>= 1; u >= 1; u >>= 1) seg[u] = merge_node(seg[2 * u], seg[2 * u + 1]);
        } else {
            int l, r;
            cin >> l >> r;
            Node L{0, NEG, NEG, NEG}, R{0, NEG, NEG, NEG};
            bool hasL = false, hasR = false;
            for (l += sz - 1, r += sz; l < r; l >>= 1, r >>= 1) {
                if (l & 1) { L = hasL ? merge_node(L, seg[l]) : seg[l]; hasL = true; l++; }
                if (r & 1) { --r; R = hasR ? merge_node(seg[r], R) : seg[r]; hasR = true; }
            }
            Node res = !hasL ? R : (!hasR ? L : merge_node(L, R));
            cout << res.best << '\n';
        }
    }
    return 0;
}
