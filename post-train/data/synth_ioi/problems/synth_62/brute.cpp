#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    vector<int> a(n);
    for (int i = 0; i < n; i++) cin >> a[i];
    vector<int> best(n, INT_MAX);
    vector<char> used(n, 0);
    best[0] = 0;
    long long total = 0;
    for (int it = 0; it < n; it++) {
        int u = -1;
        for (int i = 0; i < n; i++) {
            if (!used[i] && (u == -1 || best[i] < best[u])) u = i;
        }
        used[u] = 1;
        total += best[u];
        int au = a[u];
        for (int v = 0; v < n; v++) if (!used[v]) {
            int w = au ^ a[v];
            if (w < best[v]) best[v] = w;
        }
    }
    cout << total << '\n';
    return 0;
}
