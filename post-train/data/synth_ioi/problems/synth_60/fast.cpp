#include <bits/stdc++.h>
using namespace std;

static constexpr long long MOD = 1'000'000'007LL;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];

    vector<long long> rmax(n), rmin(n), ps_max(n + 1), ps_min(n + 1), ps_prod(n + 1);
    long long ans = 0;

    function<void(int, int)> solve = [&](int lo, int hi) {
        if (lo == hi) {
            long long v = a[lo] % MOD;
            ans = (ans + v * v) % MOD;
            return;
        }
        int mid = (lo + hi) >> 1;
        solve(lo, mid);
        solve(mid + 1, hi);

        int rlen = hi - mid;
        ps_max[0] = ps_min[0] = ps_prod[0] = 0;
        long long cur_mx = 0, cur_mn = (1LL << 60);
        for (int j = mid + 1; j <= hi; j++) {
            cur_mx = max(cur_mx, a[j]);
            cur_mn = min(cur_mn, a[j]);
            int idx = j - mid; // 1 .. rlen
            rmax[idx - 1] = cur_mx;
            rmin[idx - 1] = cur_mn;
            long long mxm = cur_mx % MOD, mnm = cur_mn % MOD;
            ps_max[idx] = (ps_max[idx - 1] + mxm) % MOD;
            ps_min[idx] = (ps_min[idx - 1] + mnm) % MOD;
            ps_prod[idx] = (ps_prod[idx - 1] + mxm * mnm) % MOD;
        }

        long long lmx = 0, lmn = (1LL << 60);
        int p = 0, q = 0; // 0-indexed into rmax/rmin [0 .. rlen-1]
        for (int i = mid; i >= lo; i--) {
            lmx = max(lmx, a[i]);
            lmn = min(lmn, a[i]);
            while (p < rlen && rmax[p] <= lmx) p++;
            while (q < rlen && rmin[q] >= lmn) q++;

            long long lmx_m = lmx % MOD, lmn_m = lmn % MOD;
            int u = min(p, q), v = max(p, q);
            // [0, u): both lmx and lmn dominate
            long long ways = (long long)u * ((lmx_m * lmn_m) % MOD) % MOD;
            // [u, v): one side from left, one from right
            if (p < q) {
                long long s = (ps_max[q] - ps_max[p] + MOD) % MOD;
                ways = (ways + lmn_m * s) % MOD;
            } else if (q < p) {
                long long s = (ps_min[p] - ps_min[q] + MOD) % MOD;
                ways = (ways + lmx_m * s) % MOD;
            }
            // [v, rlen): both rmax and rmin dominate
            long long s2 = (ps_prod[rlen] - ps_prod[v] + MOD) % MOD;
            ways = (ways + s2) % MOD;
            ans += ways;
            if (ans >= MOD) ans -= MOD;
        }
    };

    solve(0, n - 1);
    cout << ans % MOD << '\n';
    return 0;
}
