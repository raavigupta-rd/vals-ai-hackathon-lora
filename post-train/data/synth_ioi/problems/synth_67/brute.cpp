#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    long long ans = 0;
    for (int j = 0; j < n; j++) {
        int d = a[j];
        for (int i = 0; i < n; i++) ans += a[i] / d;
    }
    cout << ans << '\n';
    return 0;
}
