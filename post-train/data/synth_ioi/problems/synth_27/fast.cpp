#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    cin >> n;
    priority_queue<long long, vector<long long>, greater<>> pq;
    long long sum = 0;
    for (int i = 1; i <= n; i++) {
        long long x;
        cin >> x;
        sum += x;
        pq.push(x);
        if (sum < 0) {
            sum -= pq.top();
            pq.pop();
        }
        cout << pq.size() << (i == n ? '\n' : ' ');
    }
    return 0;
}
