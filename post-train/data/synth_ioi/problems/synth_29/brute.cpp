#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> pg(n + 1, 0), pr(n + 1, 0);
    for (int i = 1; i <= n; i++) { long long x; cin >> x; pg[i] = pg[i - 1] + x; }
    for (int i = 1; i <= n; i++) { long long x; cin >> x; pr[i] = pr[i - 1] + x; }

    long long best = 0;
    for (int l = 1; l + k - 1 <= n; l++) {
        for (int r = l + k - 1; r <= n; r++) {
            __int128 num = (__int128)1000000LL * (pg[r] - pg[l - 1]);
            long long den = pr[r] - pr[l - 1];
            best = max(best, (long long)(num / den));
        }
    }
    cout << best << '\n';
    return 0;
}
