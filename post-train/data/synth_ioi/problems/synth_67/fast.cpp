#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    int mx = 0;
    vector<int> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
        mx = max(mx, a[i]);
    }
    vector<int> cnt(mx + 1, 0), pref(mx + 1, 0);
    for (int x : a) cnt[x]++;
    for (int v = 1; v <= mx; v++) pref[v] = pref[v - 1] + cnt[v];

    long long ans = 0;
    for (int d = 1; d <= mx; d++) {
        if (!cnt[d]) continue;
        long long cd = cnt[d];
        for (int k = 1, lo = d; lo <= mx; k++, lo += d) {
            int hi = min(mx, lo + d - 1);
            long long num = pref[hi] - pref[lo - 1];
            ans += cd * k * num;
        }
    }
    cout << ans << '\n';
    return 0;
}
