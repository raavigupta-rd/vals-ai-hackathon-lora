#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n, q;
    string s;
    cin >> n >> q >> s;
    vector<int> pi(n + 1, 0);
    for (int i = 2; i <= n; i++) {
        int j = pi[i - 1];
        while (j > 0 && s[j] != s[i - 1]) j = pi[j];
        if (s[j] == s[i - 1]) j++;
        pi[i] = j;
    }
    while (q--) {
        int a, b;
        cin >> a >> b;
        a = pi[a];
        b = pi[b];
        while (a != b) {
            if (a > b) a = pi[a];
            else b = pi[b];
        }
        cout << a << '\n';
    }
    return 0;
}
