#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<long long> x(n), y(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i];
    while (q--) {
        long long a, b;
        cin >> a >> b;
        long long best = -(1LL << 62);
        for (int i = 0; i < n; i++) best = max(best, a * x[i] + b * y[i]);
        cout << best << '\n';
    }
    return 0;
}
