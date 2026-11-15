#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, k;
    cin >> n >> k;
    vector<long long> g(n + 1), r(n + 1);
    for (int i = 1; i <= n; i++) cin >> g[i];
    for (int i = 1; i <= n; i++) cin >> r[i];

    vector<__int128> pref(n + 1, 0);
    auto ok = [&](long long X) -> bool {
        __int128 mn = 0;
        for (int i = 1; i <= n; i++) {
            pref[i] = pref[i - 1] + (__int128)1000000LL * g[i] - (__int128)X * r[i];
            if (i >= k) {
                if (pref[i - k] < mn) mn = pref[i - k];
                if (pref[i] - mn >= 0) return true;
            }
        }
        return false;
    };

    long long lo = 0, hi = 1000000000000LL;
    while (lo < hi) {
        long long mid = lo + (hi - lo + 1) / 2;
        if (ok(mid)) lo = mid;
        else hi = mid - 1;
    }
    cout << lo << '\n';
    return 0;
}
