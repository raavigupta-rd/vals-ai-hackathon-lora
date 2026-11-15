#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    cin >> n >> q;
    vector<int> h(n + 1);
    for (int i = 1; i <= n; i++) cin >> h[i];
    vector<int> dq(n + 2);
    while (q--) {
        int l, r, w;
        cin >> l >> r >> w;
        int head = 0, tail = 0;
        int best = 0;
        for (int i = l; i <= r; i++) {
            while (head < tail && h[dq[tail - 1]] >= h[i]) tail--;
            dq[tail++] = i;
            if (dq[head] <= i - w) head++;
            if (i >= l + w - 1) best = max(best, h[dq[head]]);
        }
        cout << best << '\n';
    }
    return 0;
}
