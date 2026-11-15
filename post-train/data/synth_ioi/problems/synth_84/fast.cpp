#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, K;
    cin >> n >> K;
    vector<long long> P(n + 1, 0);
    for (int i = 1; i <= n; i++) {
        long long x; cin >> x;
        P[i] = P[i - 1] + x;
    }
    if (K == n) {
        long long s = 0;
        for (int i = 1; i <= n; i++) {
            long long d = P[i] - P[i - 1];
            s += d * d;
        }
        cout << s << '\n';
        return 0;
    }

    const __int128 BIG = n + 2;
    vector<__int128> dp(n + 1, 0);
    vector<int> dq(n + 2);

    // Solve penalized CHT minimizing BIG * (sum S_t^2 + lam * k) + k
    auto solve_pen = [&](long long lam) -> pair<long long, int> {
        const __int128 pen = (__int128)lam * BIG + 1;
        auto M_of = [&](int j) -> __int128 { return -2 * BIG * P[j]; };
        auto B_of = [&](int j) -> __int128 { return dp[j] + BIG * P[j] * P[j]; };
        auto eval = [&](int j, long long x) -> __int128 { return M_of(j) * x + B_of(j); };

        dp[0] = 0;
        int head = 0, tail = 0;
        dq[tail++] = 0;

        for (int i = 1; i <= n; i++) {
            long long x = P[i];
            while (tail - head >= 2 && eval(dq[head + 1], x) <= eval(dq[head], x)) head++;
            int best_j = dq[head];
            dp[i] = eval(best_j, x) + BIG * x * x + pen;

            // Slopes M_of(j) = -2*BIG*P[j] are strictly decreasing; pop redundant back lines
            __int128 m3 = M_of(i), b3 = B_of(i);
            while (tail - head >= 2) {
                int j1 = dq[tail - 2], j2 = dq[tail - 1];
                __int128 m1 = M_of(j1), b1 = B_of(j1);
                __int128 m2 = M_of(j2), b2 = B_of(j2);
                // Intersection(j1, j2) = (b2 - b1)/(m1 - m2) >= (b3 - b2)/(m2 - m3)
                // Multiply by positive (m1 - m2)*(m2 - m3):
                // Wait: P[N] <= 3e8, BIG <= 1.5e5 -> b <= 1.5e22, m <= 9e13, b*m <= 1.35e36 < INT128_MAX (1.7e38)
                if ((b2 - b1) * (m2 - m3) >= (b3 - b2) * (m1 - m2)) tail--;
                else break;
            }
            dq[tail++] = i;
        }

        int cnt = (int)(dp[n] % BIG);
        long long f_plus_lam_k = (long long)(dp[n] / BIG);
        return {f_plus_lam_k, cnt};
    };

    long long lo = 0, hi = P[n] * P[n] + 1;
    while (lo < hi) {
        long long mid = lo + (hi - lo) / 2;
        if (solve_pen(mid).second <= K) hi = mid;
        else lo = mid + 1;
    }
    auto [f_pen, _] = solve_pen(lo);
    cout << (f_pen - (__int128)lo * K > 0 ? (long long)((__int128)f_pen - (__int128)lo * K) : 0LL) << '\n';
    return 0;
}
