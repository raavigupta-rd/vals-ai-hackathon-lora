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
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        long long mx = a[i], mn = a[i];
        for (int j = i; j < n; j++) {
            if (a[j] > mx) mx = a[j];
            if (a[j] < mn) mn = a[j];
            ans = (ans + (mx % MOD) * (mn % MOD)) % MOD;
        }
    }
    cout << ans << '\n';
    return 0;
}
