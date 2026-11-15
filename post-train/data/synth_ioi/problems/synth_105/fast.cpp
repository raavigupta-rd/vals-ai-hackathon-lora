#include <bits/stdc++.h>
using namespace std;

using u64 = unsigned long long;

static inline u64 splitmix1(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline u64 splitmix2(u64 x) {
    x += 0x243f6a8885a308d3ULL;
    x = (x ^ (x >> 29)) * 0x13198a2e03707344ULL;
    x = (x ^ (x >> 26)) * 0xa4093822299f31d0ULL;
    return x ^ (x >> 32);
}

struct Hash128 {
    u64 h1, h2;
    bool operator<(const Hash128& o) const {
        return h1 != o.h1 ? h1 < o.h1 : h2 < o.h2;
    }
    bool operator==(const Hash128& o) const {
        return h1 == o.h1 && h2 == o.h2;
    }
};

static inline Hash128 wrap_child(const Hash128& c, int sz) {
    u64 a = splitmix1(c.h1 ^ (c.h2 >> 13) ^ ((u64)sz * 0x517cc1b727220a95ULL));
    u64 b = splitmix2(c.h2 ^ (c.h1 >> 11) ^ ((u64)sz * 0x9e3779b97f4a7c15ULL));
    return {a, b};
}

static inline Hash128 finalize_sum(u64 s1, u64 s2, int sz) {
    u64 a = splitmix1(s1 + 0x6a09e667f3bcc909ULL + (u64)sz);
    u64 b = splitmix2(s2 + 0xbb67ae8584caa73bULL + (u64)sz * 3ULL);
    return {a, b};
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n)) return 0;
    vector<vector<int>> adj(n + 1);
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        cin >> u >> v;
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    vector<int> par(n + 1, 0), ord, sz(n + 1, 1);
    ord.reserve(n);
    ord.push_back(1);
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        for (int v : adj[u]) {
            if (v != par[u]) {
                par[v] = u;
                ord.push_back(v);
            }
        }
    }

    vector<u64> sum1(n + 1, 0), sum2(n + 1, 0);
    vector<Hash128> down(n + 1), root_hash(n + 1);
    for (int i = n - 1; i >= 0; i--) {
        int u = ord[i];
        u64 s1 = 0, s2 = 0;
        for (int v : adj[u]) {
            if (v == par[u]) continue;
            sz[u] += sz[v];
            Hash128 w = wrap_child(down[v], sz[v]);
            s1 += w.h1;
            s2 += w.h2;
        }
        sum1[u] = s1;
        sum2[u] = s2;
        down[u] = finalize_sum(s1, s2, sz[u]);
    }

    vector<Hash128> up_wrapped(n + 1, {0, 0});
    for (int i = 0; i < n; i++) {
        int u = ord[i];
        u64 tot1 = sum1[u] + up_wrapped[u].h1;
        u64 tot2 = sum2[u] + up_wrapped[u].h2;
        root_hash[u] = finalize_sum(tot1, tot2, n);
        for (int v : adj[u]) {
            if (v == par[u]) continue;
            Hash128 wv = wrap_child(down[v], sz[v]);
            u64 rem1 = tot1 - wv.h1;
            u64 rem2 = tot2 - wv.h2;
            Hash128 up_sub = finalize_sum(rem1, rem2, n - sz[v]);
            up_wrapped[v] = wrap_child(up_sub, n - sz[v]);
        }
    }

    map<Hash128, int> id_map;
    vector<int> ans(n + 1);
    int next_id = 0;
    for (int u = 1; u <= n; u++) {
        auto [it, inserted] = id_map.emplace(root_hash[u], next_id + 1);
        if (inserted) ++next_id;
        ans[u] = it->second;
    }

    cout << next_id << '\n';
    for (int u = 1; u <= n; u++) {
        cout << ans[u] << (u == n ? '\n' : ' ');
    }
    return 0;
}
