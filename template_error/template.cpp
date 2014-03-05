#include <iostream>

/* template test */


template< const int A >
class Base {
	public:
		template< class C >
		C getP() { return static_cast<C>(p); }
	protected:
		float p;
};

template< const int B >
class Child : public Base< B > {
	public:
	  void member() {
		this->p = 5.f;
		// this crashes with:
		//    error: expected primary-expression before ‘double’
		//Base<B>::getP<double>();
		
		// this crashes with:
		//    error: no matching function for call to ‘Child<4>::getP()’
		//Base<B>::getP();
		
		// this crashes with:
		//    error: expected primary-expression before ‘double’
		//this->getP<double>();
		
		// thats how to overcome the issue:
		std::cout << "Private float p: " << Base<B>::template getP<double>();
		std::cout << "Private float p: " << this->template getP<double>();
	  }
};



int main () {
  Child<4> myChild;
  myChild.member();
  
  return 0;  
}
