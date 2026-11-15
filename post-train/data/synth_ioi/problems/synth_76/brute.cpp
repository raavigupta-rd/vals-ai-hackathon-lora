#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<long long> x(n), h(n);
    for (int i = 0; i < n; i++) cin >> x[i] >> h[i];
    for (int i = 0; i < n; i++) {
        long long lim = x[i] + h[i];
        int j = i;
        while (j + 1 < n && x[j + 1] <= lim) {
            j++;
            lim = max(lim, x[j] + h[j]);
        }
        cout << (j - i + 1) << (i + 1 == n ? '\n' : ' ');
    }
    return 0;
}
