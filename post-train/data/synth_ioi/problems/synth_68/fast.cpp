#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    const int V = 300000;
    vector<int> freq(V + 1, 0);
    int mx = 0;
    for (int i = 0; i < n; i++) {
        int x; cin >> x;
        freq[x]++;
        mx = max(mx, x);
    }

    vector<long long> ways(mx + 1, 0);
    for (int d = 1; d <= mx; d++) {
        long long c = 0;
        for (int m = d; m <= mx; m += d) c += freq[m];
        if (c >= 3) ways[d] = c * (c - 1) * (c - 2) / 6;
    }
    for (int d = mx; d >= 1; d--) {
        for (int m = d + d; m <= mx; m += d) ways[d] -= ways[m];
    }

    while (q--) {
        int g; cin >> g;
        cout << (g <= mx ? ways[g] : 0LL) << '\n';
    }
    return 0;
}
