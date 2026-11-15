#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    priority_queue<long long> pq;
    long long ans = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        x -= i;
        pq.push(x);
        pq.push(x);
        ans += pq.top() - x;
        pq.pop();
    }
    cout << ans << '\n';
    return 0;
}
