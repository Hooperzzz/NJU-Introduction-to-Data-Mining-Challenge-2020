#include<bits/stdc++.h>
#define GET_INPUT_NUM cerr << INPUT_NUM << "#"
using namespace std;

int INPUT_NUM = 0;

int SCANF(const char *fmt , ...){
    int input_len = strlen(fmt);
    for(int i=0; i < input_len; i++){
        if(fmt[i] == '%')
            INPUT_NUM++;
    }

	int ret;
	va_list ap;
	va_start(ap , fmt);
	ret = vscanf(fmt,ap);
	va_end(ap);
	GET_INPUT_NUM;
	return ret;
}

#define CIN_OPERATOR(Type) \
    friend CIN &operator>>(CIN &in, Type obj){ \
        INPUT_NUM++; \
        cin >> obj; \
        GET_INPUT_NUM; \
        return in; \
    }

class CIN{
	CIN_OPERATOR(int&);
	CIN_OPERATOR(unsigned&);
	CIN_OPERATOR(short&);
	CIN_OPERATOR(unsigned short&);
	CIN_OPERATOR(long&);
	CIN_OPERATOR(unsigned long&);
	CIN_OPERATOR(long long&);
	CIN_OPERATOR(unsigned long long&);
	CIN_OPERATOR(char&);
	CIN_OPERATOR(unsigned char&);
	CIN_OPERATOR(float&);
	CIN_OPERATOR(double&);
	CIN_OPERATOR(long double&);
	CIN_OPERATOR(bool&);
	CIN_OPERATOR(string&);
	CIN_OPERATOR(char*);

public:

	operator void *() const{
	    return (void*)&cin;
	}

	bool operator!() const{
	    return !cin;
	}

	istream& getline(char* s){
	    INPUT_NUM++;
	    GET_INPUT_NUM;
	    return cin.getline(s, 10000);
	}

	istream& getline(char* s, streamsize n){
	    INPUT_NUM++;
	    GET_INPUT_NUM;
	    return cin.getline(s, n);
	}

	istream& getline(char* s, streamsize n, char delim){
	    INPUT_NUM++;
	    GET_INPUT_NUM;
	    return cin.getline(s, n, delim);
	}

	char get(){
	    INPUT_NUM++;
	    GET_INPUT_NUM;
	    return cin.get();
	}

	void get(char &a){
	    INPUT_NUM++;
	    GET_INPUT_NUM;
	    cin.get(a);
	}
} CIN_OBJ;

#define scanf SCANF
#define cin CIN_OBJ
