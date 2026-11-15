#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> p(n);
    for (int i = 0; i < n; i++) cin >> p[i];
    long long ans = 0;
    for (int i = 0; i < n; i++) {
        int mx = p[i], mn = p[i];
        for (int j = i; j < n; j++) {
            if (p[j] > mx) mx = p[j];
            if (p[j] < mn) mn = p[j];
            if (mx - mn == j - i) ans++;
        }
    }
    cout << ans << '\n';
    return 0;
}
