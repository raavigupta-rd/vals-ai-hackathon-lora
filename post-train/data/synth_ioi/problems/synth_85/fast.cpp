#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

struct Node {
    int ls = 0, rs = 0;
    long long sw = 0, sq = 0;
};

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> col(n + 1);
    vector<long long> w(n + 1);
    for (int i = 1; i <= n; i++) cin >> col[i];
    for (int i = 1; i <= n; i++) { cin >> w[i]; w[i] %= MOD; }

    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    struct Qry { int l, r, id; };
    vector<vector<Qry>> qs(n + 1);
    for (int i = 0; i < q; i++) {
        int u, l, r; cin >> u >> l >> r;
        qs[u].push_back({l, r, i});
    }

    vector<int> par(n + 1, 0), ord = {1};
    ord.reserve(n);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ord.push_back(v);
        }
    }

    vector<Node> tr;
    tr.reserve(n * 20 + 5);
    tr.push_back({}); // 0 = null

    auto new_node = [&]() -> int {
        tr.push_back({});
        return (int)tr.size() - 1;
    };

    function<int(int, int, int, long long)> make_single =
        [&](int l, int r, int pos, long long val) -> int {
            int u = new_node();
            if (l == r) {
                tr[u].sw = val;
                tr[u].sq = val * val % MOD;
                return u;
            }
            int mid = (l + r) >> 1;
            if (pos <= mid) tr[u].ls = make_single(l, mid, pos, val);
            else tr[u].rs = make_single(mid + 1, r, pos, val);
            tr[u].sq = (tr[tr[u].ls].sq + tr[tr[u].rs].sq) % MOD;
            return u;
        };

    function<int(int, int, int, int)> merge_tr =
        [&](int a, int b, int l, int r) -> int {
            if (!a || !b) return a ? a : b;
            if (l == r) {
                tr[a].sw = (tr[a].sw + tr[b].sw) % MOD;
                tr[a].sq = tr[a].sw * tr[a].sw % MOD;
                return a;
            }
            int mid = (l + r) >> 1;
            tr[a].ls = merge_tr(tr[a].ls, tr[b].ls, l, mid);
            tr[a].rs = merge_tr(tr[a].rs, tr[b].rs, mid + 1, r);
            tr[a].sq = (tr[tr[a].ls].sq + tr[tr[a].rs].sq) % MOD;
            return a;
        };

    function<long long(int, int, int, int, int)> query_tr =
        [&](int u, int l, int r, int ql, int qr) -> long long {
            if (!u || ql <= l && r <= qr) return tr[u].sq;
            int mid = (l + r) >> 1;
            long long res = 0;
            if (ql <= mid) res += query_tr(tr[u].ls, l, mid, ql, qr);
            if (qr > mid) res += query_tr(tr[u].rs, mid + 1, r, ql, qr);
            return res % MOD;
        };

    vector<int> root(n + 1, 0);
    vector<long long> ans(q, 0);

    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        root[u] = make_single(1, n, col[u], w[u]);
        for (int v : adj[u]) if (v != par[u]) {
            root[u] = merge_tr(root[u], root[v], 1, n);
        }
        for (const auto& qr : qs[u]) {
            ans[qr.id] = query_tr(root[u], 1, n, qr.l, qr.r);
        }
    }

    for (int i = 0; i < q; i++) cout << ans[i] << '\n';
    return 0;
}
