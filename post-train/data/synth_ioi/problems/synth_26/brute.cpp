#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int K, m, q;
    cin >> K >> m >> q;
    vector<int> reach(K + 2, 0);
    for (int i = 0; i < m; i++) {
        int l, r;
        cin >> l >> r;
        reach[l] = max(reach[l], r + 1);
    }
    for (int x = 1; x <= K + 1; x++) reach[x] = max(reach[x], reach[x - 1]);
    while (q--) {
        int a, b;
        cin >> a >> b;
        int cur = a, steps = 0;
        while (cur <= b) {
            if (reach[cur] <= cur) { steps = -1; break; }
            cur = reach[cur];
            steps++;
        }
        cout << steps << '\n';
    }
    return 0;
}
