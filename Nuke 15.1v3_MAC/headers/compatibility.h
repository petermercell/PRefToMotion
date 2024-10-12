#ifndef COMPATIBILITY_H
#define COMPATIBILITY_H

#if __cplusplus < 201402L
namespace std {
    template<typename T>
    using underlying_type_t = typename underlying_type<T>::type;

    template<bool B, typename T = void>
    using enable_if_t = typename enable_if<B, T>::type;
}
#endif

#endif // COMPATIBILITY_H
