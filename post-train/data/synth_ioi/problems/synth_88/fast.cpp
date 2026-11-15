#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;

    map<pair<int, int>, int> open_at;
    vector<char> is_q(q, 0);
    struct Span { int l, r, u, v; };
    vector<Span> spans;
    spans.reserve(q);

    for (int t = 0; t < q; t++) {
        int tp; cin >> tp;
        if (tp == 1) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            open_at[{u, v}] = t;
        } else if (tp == 2) {
            int u, v; cin >> u >> v;
            if (u > v) swap(u, v);
            auto it = open_at.find({u, v});
            spans.push_back({it->second, t - 1, u, v});
            open_at.erase(it);
        } else {
            is_q[t] = 1;
        }
    }
    for (auto& [e, start_t] : open_at) {
        spans.push_back({start_t, q - 1, e.first, e.second});
    }

    vector<vector<pair<int, int>>> seg(4 * q + 5);
    function<void(int, int, int, int, int, pair<int, int>)> add_span =
        [&](int node, int l, int r, int ql, int qr, pair<int, int> e) {
            if (ql <= l && r <= qr) {
                seg[node].push_back(e);
                return;
            }
            int mid = (l + r) >> 1;
            if (ql <= mid) add_span(node << 1, l, mid, ql, qr, e);
            if (qr > mid) add_span(node << 1 | 1, mid + 1, r, ql, qr, e);
        };
    for (const auto& s : spans) {
        if (s.l <= s.r) add_span(1, 0, q - 1, s.l, s.r, {s.u, s.v});
    }

    vector<int> p(n + 1), sz(n + 1, 1);
    iota(p.begin(), p.end(), 0);
    long long pairs = n;
    struct Undo { int b, a; long long prev_pairs; };
    vector<Undo> stk;
    stk.reserve(q * 2);

    auto find_root = [&](int x) {
        while (p[x] != x) x = p[x];
        return x;
    };
    auto unite = [&](int u, int v) {
        int a = find_root(u), b = find_root(v);
        if (a == b) return;
        if (sz[a] < sz[b]) swap(a, b);
        stk.push_back({b, a, pairs});
        pairs += 2LL * sz[a] * sz[b];
        p[b] = a;
        sz[a] += sz[b];
    };

    vector<long long> ans(q, -1);
    function<void(int, int, int)> dfs = [&](int node, int l, int r) {
        size_t snap = stk.size();
        for (auto [u, v] : seg[node]) unite(u, v);
        if (l == r) {
            if (is_q[l]) ans[l] = pairs;
        } else {
            int mid = (l + r) >> 1;
            dfs(node << 1, l, mid);
            dfs(node << 1 | 1, mid + 1, r);
        }
        while (stk.size() > snap) {
            auto un = stk.back(); stk.pop_back();
            sz[un.a] -= sz[un.b];
            p[un.b] = un.b;
            pairs = un.prev_pairs;
        }
    };

    dfs(1, 0, q - 1);
    for (int t = 0; t < q; t++) if (is_q[t]) cout << ans[t] << '\n';
    return 0;
}
