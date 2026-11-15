#include <bits/stdc++.h>
using namespace std;

static constexpr long long NEG_INF = -(1LL << 60);

struct Mat {
    long long m[2][2];
    Mat() {
        m[0][0] = m[1][1] = 0;
        m[0][1] = m[1][0] = NEG_INF;
    }
};

static inline Mat mul(const Mat& A, const Mat& B) {
    Mat C;
    C.m[0][0] = max(A.m[0][0] + B.m[0][0], A.m[0][1] + B.m[1][0]);
    C.m[0][1] = max(A.m[0][0] + B.m[0][1], A.m[0][1] + B.m[1][1]);
    C.m[1][0] = max(A.m[1][0] + B.m[0][0], A.m[1][1] + B.m[1][0]);
    C.m[1][1] = max(A.m[1][0] + B.m[0][1], A.m[1][1] + B.m[1][1]);
    return C;
}

static inline Mat make_node_mat(long long g0, long long g1) {
    Mat M;
    M.m[0][0] = g0; M.m[0][1] = g0;
    M.m[1][0] = g1; M.m[1][1] = NEG_INF;
    return M;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> a(n + 1);
    for (int i = 1; i <= n; i++) cin >> a[i];
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v; cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), sz(n + 1, 1), heavy(n + 1, 0), ord = {1};
    ord.reserve(n);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) if (v != par[u]) {
            par[v] = u;
            ord.push_back(v);
        }
    }
    for (int i = n - 1; i >= 1; i--) {
        int u = ord[i];
        sz[par[u]] += sz[u];
        if (sz[u] > sz[heavy[par[u]]]) heavy[par[u]] = u;
    }

    vector<int> head(n + 1, 0), tail(n + 1, 0), pos(n + 1, 0), inv_pos(n + 1, 0);
    int timer = 0;
    {
        vector<pair<int, int>> st = {{1, 1}};
        while (!st.empty()) {
            auto [u, h] = st.back(); st.pop_back();
            int last = u;
            for (int x = u; x != 0; x = heavy[x]) {
                head[x] = h;
                pos[x] = ++timer;
                inv_pos[timer] = x;
                last = x;
                for (int v : adj[x]) if (v != par[x] && v != heavy[x]) st.push_back({v, v});
            }
            tail[h] = last;
        }
    }

    vector<long long> dp0(n + 1, 0), dp1(n + 1, 0), g0(n + 1, 0), g1(n + 1, 0);
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        g1[u] = a[u];
        for (int v : adj[u]) if (v != par[u] && v != heavy[u]) {
            g0[u] += max(dp0[v], dp1[v]);
            g1[u] += dp0[v];
        }
        dp0[u] = g0[u];
        dp1[u] = g1[u];
        if (heavy[u]) {
            dp0[u] += max(dp0[heavy[u]], dp1[heavy[u]]);
            dp1[u] += dp0[heavy[u]];
        }
    }

    int S = 1;
    while (S < n) S <<= 1;
    vector<Mat> seg(2 * S);
    for (int i = 1; i <= n; i++) {
        int u = inv_pos[i];
        seg[S + i - 1] = make_node_mat(g0[u], g1[u]);
    }
    for (int i = S - 1; i >= 1; i--) seg[i] = mul(seg[2 * i], seg[2 * i + 1]);

    auto seg_set = [&](int idx, const Mat& M) {
        int u = S + idx - 1;
        seg[u] = M;
        for (u >>= 1; u >= 1; u >>= 1) seg[u] = mul(seg[2 * u], seg[2 * u + 1]);
    };
    auto seg_query = [&](int l, int r) {
        Mat L, R;
        for (l += S - 1, r += S; l < r; l >>= 1, r >>= 1) {
            if (l & 1) L = mul(L, seg[l++]);
            if (r & 1) R = mul(seg[--r], R);
        }
        return mul(L, R);
    };
    auto get_chain_dp = [&](int h) -> pair<long long, long long> {
        Mat P = seg_query(pos[h], pos[tail[h]]);
        // Applied to base column [0, NEG_INF]^T -> [P.m[0][0], P.m[1][0]]
        return {P.m[0][0], P.m[1][0]};
    };

    while (q--) {
        int u; long long nx;
        cin >> u >> nx;
        g1[u] += nx - a[u];
        a[u] = nx;

        while (u != 0) {
            int h = head[u];
            int p = par[h];
            pair<long long, long long> old_dp = {0, 0};
            if (p != 0) old_dp = {dp0[h], dp1[h]};

            seg_set(pos[u], make_node_mat(g0[u], g1[u]));
            auto new_dp = get_chain_dp(h);
            dp0[h] = new_dp.first;
            dp1[h] = new_dp.second;

            if (p != 0) {
                g0[p] += max(new_dp.first, new_dp.second) - max(old_dp.first, old_dp.second);
                g1[p] += new_dp.first - old_dp.first;
            }
            u = p;
        }
        cout << max(dp0[1], dp1[1]) << '\n';
    }
    return 0;
}
