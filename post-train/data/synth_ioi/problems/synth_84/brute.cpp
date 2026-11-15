#include <bits/stdc++.h>
using namespace std;

// 40-pt subtask solver: O(N * K) Monotonic-Deque Convex Hull Trick DP (O(N) memory)
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
    const long long INF = (1LL << 61);
    vector<long long> prv(n + 1, INF), cur(n + 1, INF);
    prv[0] = 0;
    vector<int> dq(n + 2);

    for (int t = 1; t <= K; t++) {
        auto M_of = [&](int j) { return -2LL * P[j]; };
        auto B_of = [&](int j) { return prv[j] + P[j] * P[j]; };
        auto eval = [&](int j, long long x) { return M_of(j) * x + B_of(j); };

        int head = 0, tail = 0;
        dq[tail++] = t - 1;
        for (int i = t; i <= n; i++) {
            long long x = P[i];
            while (tail - head >= 2 && eval(dq[head + 1], x) <= eval(dq[head], x)) head++;
            cur[i] = eval(dq[head], x) + x * x;

            if (prv[i] < INF) {
                long long m3 = M_of(i), b3 = B_of(i);
                while (tail - head >= 2) {
                    int j1 = dq[tail - 2], j2 = dq[tail - 1];
                    __int128 lhs = (__int128)(B_of(j2) - B_of(j1)) * (M_of(j2) - m3);
                    __int128 rhs = (__int128)(b3 - B_of(j2)) * (M_of(j1) - M_of(j2));
                    if (lhs >= rhs) tail--;
                    else break;
                }
                dq[tail++] = i;
            }
        }
        fill(prv.begin(), prv.begin() + t, INF);
        for (int i = t; i <= n; i++) prv[i] = cur[i];
    }
    cout << prv[n] << '\n';
    return 0;
}
