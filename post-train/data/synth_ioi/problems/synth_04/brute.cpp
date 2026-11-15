#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> x(n), y(n);
    vector<long long> w(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> y[i] >> w[i];
    while (q--) {
        int x1, y1, x2, y2;
        cin >> x1 >> y1 >> x2 >> y2;
        long long s = 0;
        for (int i = 0; i < n; i++) {
            if (x1 <= x[i] && x[i] <= x2 && y1 <= y[i] && y[i] <= y2) s += w[i];
        }
        cout << s << '\n';
    }
    return 0;
}
