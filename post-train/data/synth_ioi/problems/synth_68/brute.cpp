#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> a(n);
    int mx = 0;
    for (int i = 0; i < n; i++) { cin >> a[i]; mx = max(mx, a[i]); }
    vector<long long> ways(mx + 1, 0);
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            int g2 = std::gcd(a[i], a[j]);
            for (int k = j + 1; k < n; k++) {
                ways[std::gcd(g2, a[k])]++;
            }
        }
    }
    while (q--) {
        int g; cin >> g;
        cout << (g <= mx ? ways[g] : 0LL) << '\n';
    }
    return 0;
}
