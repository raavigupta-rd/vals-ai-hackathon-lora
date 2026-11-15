#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k, L;
    cin >> n >> k >> L;
    vector<long long> pref(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        pref[i] = pref[i - 1] + x;
    }
    vector<long long> dp_val(n + 1, 0);
    vector<int> dp_cnt(n + 1, 0);

    // Solve penalized DP: maximize (sum - lam * count), breaking ties toward MINIMUM count.
    auto solve_pen = [&](long long lam) -> pair<long long, int> {
        dp_val[0] = 0;
        dp_cnt[0] = 0;
        for (int i = 1; i <= n; i++) {
            dp_val[i] = dp_val[i - 1];
            dp_cnt[i] = dp_cnt[i - 1];
            if (i >= L) {
                long long cand_v = dp_val[i - L] + (pref[i] - pref[i - L]) - lam;
                int cand_c = dp_cnt[i - L] + 1;
                if (cand_v > dp_val[i] || (cand_v == dp_val[i] && cand_c < dp_cnt[i])) {
                    dp_val[i] = cand_v;
                    dp_cnt[i] = cand_c;
                }
            }
        }
        return {dp_val[n], dp_cnt[n]};
    };

    auto [v0, c0] = solve_pen(0);
    if (c0 <= k) {
        cout << v0 << '\n';
        return 0;
    }

    long long mx_w = 0;
    for (int i = L; i <= n; i++) mx_w = max(mx_w, pref[i] - pref[i - L]);
    long long lo = 1, hi = mx_w + 1;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (solve_pen(mid).second <= k) hi = mid;
        else lo = mid + 1;
    }
    auto [v_opt, _] = solve_pen(lo);
    cout << v_opt + lo * 1LL * k << '\n';
    return 0;
}
