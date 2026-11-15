#include <bits/stdc++.h>
using namespace std;

static const long long MOD = 998244353;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    long long K;
    if (!(cin >> n >> K)) return 0;
    vector<vector<pair<int, long long>>> adj(n + 1);
    long long max_edge = 0;
    for (int i = 0; i < n - 1; i++) {
        int u, v;
        long long w;
        cin >> u >> v >> w;
        adj[u].push_back({v, w});
        adj[v].push_back({u, w});
        max_edge = max(max_edge, w);
    }

    vector<vector<long long>> cent_dists, sub_dists;
    cent_dists.reserve(n);
    sub_dists.reserve(n);

    vector<char> dead(n + 1, 0);
    vector<int> sz(n + 1, 0), par_t(n + 1, 0), nodes;
    vector<long long> d_tmp(n + 1, 0);
    nodes.reserve(n);

    auto build_cd = [&](auto& self, int entry) -> void {
        nodes.clear();
        par_t[entry] = 0;
        nodes.push_back(entry);
        for (size_t i = 0; i < nodes.size(); i++) {
            int u = nodes[i];
            sz[u] = 1;
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_t[u]) {
                    par_t[v] = u;
                    nodes.push_back(v);
                }
            }
        }
        int tot = (int)nodes.size();
        if (tot <= 1) {
            dead[entry] = 1;
            return;
        }
        for (int i = tot - 1; i >= 0; i--) {
            int u = nodes[i];
            if (par_t[u]) sz[par_t[u]] += sz[u];
        }
        int cent = entry;
        for (int u : nodes) {
            int mx = tot - sz[u];
            for (auto [v, w] : adj[u]) {
                if (!dead[v] && v != par_t[u]) mx = max(mx, sz[v]);
            }
            if (mx * 2 <= tot) {
                cent = u;
                break;
            }
        }

        dead[cent] = 1;
        vector<long long> all_c;
        all_c.reserve(tot);
        all_c.push_back(0);

        for (auto [nb, w0] : adj[cent]) {
            if (dead[nb]) continue;
            nodes.clear();
            par_t[nb] = cent;
            d_tmp[nb] = w0;
            nodes.push_back(nb);
            for (size_t i = 0; i < nodes.size(); i++) {
                int u = nodes[i];
                for (auto [v, w] : adj[u]) {
                    if (!dead[v] && v != par_t[u]) {
                        par_t[v] = u;
                        d_tmp[v] = d_tmp[u] + w;
                        nodes.push_back(v);
                    }
                }
            }
            vector<long long> sv;
            sv.reserve(nodes.size());
            for (int u : nodes) {
                sv.push_back(d_tmp[u]);
                all_c.push_back(d_tmp[u]);
            }
            sort(sv.begin(), sv.end());
            if (sv.size() >= 2) sub_dists.push_back(move(sv));
        }
        sort(all_c.begin(), all_c.end());
        if (all_c.size() >= 2) cent_dists.push_back(move(all_c));

        for (auto [nb, w0] : adj[cent]) {
            if (!dead[nb]) self(self, nb);
        }
    };

    build_cd(build_cd, 1);

    auto count_pairs = [&](long long X) -> long long {
        long long total = 0;
        for (const auto& vec : cent_dists) {
            int r = (int)vec.size() - 1;
            for (int l = 0; l < r; l++) {
                while (r > l && vec[l] + vec[r] > X) r--;
                if (r > l) total += (r - l);
            }
        }
        for (const auto& vec : sub_dists) {
            int r = (int)vec.size() - 1;
            for (int l = 0; l < r; l++) {
                while (r > l && vec[l] + vec[r] > X) r--;
                if (r > l) total -= (r - l);
            }
        }
        return total;
    };

    long long lo = 1, hi = max_edge * (n - 1), kth_d = hi;
    while (lo <= hi) {
        long long mid = lo + (hi - lo) / 2;
        if (count_pairs(mid) >= K) {
            kth_d = mid;
            hi = mid - 1;
        } else {
            lo = mid + 1;
        }
    }

    // Sum and count strictly less than kth_d
    auto count_and_sum = [&](const vector<vector<long long>>& lists, long long X) -> pair<long long, long long> {
        long long cnt = 0, sm = 0;
        vector<long long> pref;
        for (const auto& vec : lists) {
            int m = (int)vec.size();
            pref.resize(m + 1);
            pref[0] = 0;
            for (int i = 0; i < m; i++) pref[i + 1] = (pref[i] + vec[i]) % MOD;
            int r = m - 1;
            for (int l = 0; l < r; l++) {
                while (r > l && vec[l] + vec[r] > X) r--;
                if (r > l) {
                    long long c = r - l;
                    cnt += c;
                    long long range_sum = (pref[r + 1] - pref[l + 1] + MOD) % MOD;
                    sm = (sm + c % MOD * (vec[l] % MOD) + range_sum) % MOD;
                }
            }
        }
        return {cnt, sm};
    };

    auto [c1, s1] = count_and_sum(cent_dists, kth_d - 1);
    auto [c2, s2] = count_and_sum(sub_dists, kth_d - 1);
    long long strict_cnt = c1 - c2;
    long long strict_sum = (s1 - s2 + MOD) % MOD;
    long long rem = (K - strict_cnt) % MOD;
    long long ans_sum = (strict_sum + rem * (kth_d % MOD)) % MOD;
    cout << kth_d << ' ' << ans_sum << '\n';
    return 0;
}
