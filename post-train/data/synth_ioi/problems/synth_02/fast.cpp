#include <bits/stdc++.h>
using namespace std;

struct Node {
    int cand = 0, cnt = 0;
};

static inline Node merge_node(const Node& a, const Node& b) {
    if (a.cand == b.cand) return {a.cand, a.cnt + b.cnt};
    if (a.cnt >= b.cnt) return {a.cand, a.cnt - b.cnt};
    return {b.cand, b.cnt - a.cnt};
}

struct BIT {
    vector<int> t;
    void init(int n) { t.assign(n + 1, 0); }
    void add(int i, int delta) {
        for (++i; i < (int)t.size(); i += i & -i) t[i] += delta;
    }
    int sum_prefix(int i) const { // sum over [0..i-1]
        int s = 0;
        for (; i > 0; i -= i & -i) s += t[i];
        return s;
    }
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> init_a(n + 1);
    vector<vector<int>> positions(n + 1);
    for (int i = 1; i <= n; i++) {
        cin >> init_a[i];
        positions[init_a[i]].push_back(i);
    }
    struct Op { int type, a, b; };
    vector<Op> ops(q);
    for (int i = 0; i < q; i++) {
        cin >> ops[i].type >> ops[i].a >> ops[i].b;
        if (ops[i].type == 1) positions[ops[i].b].push_back(ops[i].a);
    }
    vector<BIT> bits(n + 1);
    for (int c = 1; c <= n; c++) {
        auto& v = positions[c];
        sort(v.begin(), v.end());
        v.erase(unique(v.begin(), v.end()), v.end());
        bits[c].init((int)v.size());
    }
    auto add_color = [&](int c, int pos, int delta) {
        const auto& v = positions[c];
        int idx = (int)(lower_bound(v.begin(), v.end(), pos) - v.begin());
        bits[c].add(idx, delta);
    };
    auto count_range = [&](int c, int l, int r) -> int {
        if (c <= 0 || c > n) return 0;
        const auto& v = positions[c];
        int ri = (int)(upper_bound(v.begin(), v.end(), r) - v.begin());
        int li = (int)(lower_bound(v.begin(), v.end(), l) - v.begin());
        return bits[c].sum_prefix(ri) - bits[c].sum_prefix(li);
    };

    int sz = 1;
    while (sz < n) sz <<= 1;
    vector<Node> seg(2 * sz);
    vector<int> cur = init_a;
    for (int i = 1; i <= n; i++) {
        seg[sz + i - 1] = {cur[i], 1};
        add_color(cur[i], i, +1);
    }
    for (int i = sz - 1; i >= 1; i--) seg[i] = merge_node(seg[2 * i], seg[2 * i + 1]);

    auto seg_set = [&](int pos, int val) {
        int u = sz + pos - 1;
        seg[u] = {val, 1};
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = merge_node(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_query = [&](int l, int r) {
        Node L{}, R{};
        for (l += sz - 1, r += sz; l < r; l >>= 1, r >>= 1) {
            if (l & 1) L = merge_node(L, seg[l++]);
            if (r & 1) R = merge_node(seg[--r], R);
        }
        return merge_node(L, R).cand;
    };

    for (const auto& op : ops) {
        if (op.type == 1) {
            int p = op.a, x = op.b;
            if (cur[p] != x) {
                add_color(cur[p], p, -1);
                cur[p] = x;
                add_color(cur[p], p, +1);
                seg_set(p, x);
            }
        } else {
            int l = op.a, r = op.b;
            int cand = seg_query(l, r);
            int freq = count_range(cand, l, r);
            cout << (freq * 2 > (r - l + 1) ? cand : 0) << '\n';
        }
    }
    return 0;
}
