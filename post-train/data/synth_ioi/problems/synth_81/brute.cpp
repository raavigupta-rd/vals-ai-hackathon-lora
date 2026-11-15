#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> a(n), w(n), lo(n, 0), hi(n, 0);
    for (int i = 0; i < n; i++) cin >> a[i];
    for (int i = 0; i < n; i++) cin >> w[i];
    long long max_step = 0;
    for (int i = 1; i < n; i++) {
        cin >> lo[i] >> hi[i];
        max_step += max(llabs(lo[i]), llabs(hi[i]));
    }
    if (n == 1) {
        cout << 0 << '\n';
        return 0;
    }

    // Independent O(N * V) monotone-deque DP over feasible integer range [mnB, mxB]
    // Note: every optimal B_i lies in [min(A) - sum|step|, max(A) + sum|step|],
    // or more tightly within [min_j (A_j - dist), max_j (A_j + dist)].
    // Even tighter: any B outside [min(A) - max_local, max(A) + max_local]? Actually if all B_i
    // can be shifted toward A, let's bound reachable [L_reach, R_reach] around A!
    // Wait: if B_1 in [min_j(A_j - pref_R[j]), max_j(A_j - pref_L[j])]? Not if pref_L == pref_R!
    // Exact safe box: let prefL[i] = sum_{2..i} lo, prefR[i] = sum_{2..i} hi.
    // Optimal B_i always equals A_j + shift for some j, where shift in [prefL[i]-prefR[j], prefR[i]-prefL[j]]!
    vector<long long> pL(n, 0), pR(n, 0);
    for (int i = 1; i < n; i++) { pL[i] = pL[i - 1] + lo[i]; pR[i] = pR[i - 1] + hi[i]; }
    long long mnB = LLONG_MAX, mxB = LLONG_MIN;
    if (n <= 400) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                mnB = min(mnB, a[j] + pL[i] - pR[j]);
                mxB = max(mxB, a[j] + pR[i] - pL[j]);
            }
        }
    } else {
        long long amin = *min_element(a.begin(), a.end());
        long long amax = *max_element(a.begin(), a.end());
        mnB = amin - 5000;
        mxB = amax + 5000;
    }

    int V = (int)(mxB - mnB + 1);
    const long long INF = (1LL << 61);
    vector<long long> dp(V), ndp(V, INF);
    for (int x = 0; x < V; x++) dp[x] = w[0] * llabs((mnB + x) - a[0]);

    deque<int> dq;
    for (int i = 1; i < n; i++) {
        fill(ndp.begin(), ndp.end(), INF);
        dq.clear();
        // For target y = mnB + x, prev y' = y - step in [y - hi[i], y - lo[i]], i.e. idx' in [x - hi[i], x - lo[i]]
        long long Loff = -hi[i], Roff = -lo[i];
        int ptr = max(0LL, Loff);
        for (int x = 0; x < V; x++) {
            long long ql = x + Loff, qr = x + Roff;
            while (ptr <= qr && ptr < V) {
                if (ptr >= 0) {
                    while (!dq.empty() && dp[dq.back()] >= dp[ptr]) dq.pop_back();
                    dq.push_back(ptr);
                }
                ptr++;
            }
            while (!dq.empty() && dq.front() < ql) dq.pop_front();
            if (!dq.empty() && dp[dq.front()] < INF) {
                ndp[x] = dp[dq.front()] + w[i] * llabs((mnB + x) - a[i]);
            }
        }
        dp.swap(ndp);
    }
    cout << *min_element(dp.begin(), dp.end()) << '\n';
    return 0;
}
