#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string A, B;
    cin >> A >> B;
    int n = (int)A.size(), m = (int)B.size();
    vector<int> dp(m + 1, 0), ndp(m + 1, 0);
    long long ans = 0;
    for (int i = n - 1; i >= 0; i--) {
        fill(ndp.begin(), ndp.end(), 0);
        for (int j = m - 1; j >= 0; j--) {
            if (A[i] == B[j]) {
                ndp[j] = 1 + dp[j + 1];
                ans += ndp[j];
            }
        }
        dp.swap(ndp);
    }
    cout << ans << '\n';
    return 0;
}
