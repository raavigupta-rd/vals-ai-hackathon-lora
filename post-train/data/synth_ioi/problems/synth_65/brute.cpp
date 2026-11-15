#include <bits/stdc++.h>
using namespace std;

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    string t;
    int k;
    cin >> t >> k;
    vector<string> pats(k);
    for (int i = 0; i < k; i++) cin >> pats[i];
    // Sort patterns descending by length so first suffix match is longest
    sort(pats.begin(), pats.end(), [](const string& a, const string& b) {
        return a.size() > b.size();
    });

    string res;
    for (char ch : t) {
        res.push_back(ch);
        int rem = 0;
        for (const string& p : pats) {
            size_t m = p.size();
            if (m <= res.size() && res.compare(res.size() - m, m, p) == 0) {
                rem = (int)m;
                break;
            }
        }
        if (rem > 0) res.resize(res.size() - rem);
    }
    if (res.empty()) cout << "EMPTY\n";
    else cout << res << '\n';
    return 0;
}
