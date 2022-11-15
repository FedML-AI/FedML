/*
	libpaillier - A library implementing the Paillier cryptosystem.

	Copyright (C) 2006 SRI International.

	This program is free software; you can redistribute it and/or modify
	it under the terms of the GNU General Public License as published by
	the Free Software Foundation; either version 2 of the License, or
	(at your option) any later version.

	This program is distributed in the hope that it will be useful, but
	WITHOUT ANY WARRANTY; without even the implied warranty of
	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
	General Public License for more details.
*/

/*
	Include gmp.h before including this file.
*/

/*
	IMPORTANT SECURITY NOTES:

	On paillier_keygen and strong primes:

  When selecting a modulus n = p q, we do not bother ensuring that p
	and q are strong primes. While this was at one point considered
	important (e.g., it is required by ANSI X9.31), modern factoring
	algorithms make any advantage of strong primes doubtful [1]. RSA
	Laboratories no longer recommends the practice [2].

  On memory handling:

  At no point is any special effort made to securely "shred" sensitive
  memory or prevent it from being paged out to disk. This means that
  it is important that functions dealing with private keys and
  plaintexts (e.g., paillier_keygen and paillier_enc) only be run on
  trusted machines. The resulting ciphertexts and public keys,
  however, may of course be handled in an untrusted manner.

  [1] Are "strong" primes needed for RSA? Ron Rivest and Robert
      Silverman. Cryptology ePrint Archive, Report 2001/007, 2001.

  [2] RSA Laboratories' Frequently Asked Questions About Today's
      Cryptography, Version 4.1, Section 3.1.4.
*/

/******
 TYPES
*******/

/*
	This represents a Paillier public key, which is basically just the
	modulus n. The other values should be considered private.
*/
typedef struct
{
	int bits;  /* e.g., 1024 */
	mpz_t n;   /* public modulus n = p q */
	mpz_t n_squared; /* cached to avoid recomputing */
	mpz_t n_plusone; /* cached to avoid recomputing */
} paillier_pubkey_t;

/*
  This represents a Paillier private key; it needs to be used with a
  paillier_pubkey_t to be meaningful. It only includes the Carmichael
  function (lambda) of the modulus. The other value is kept for
  efficiency and should be considered private.
*/
typedef struct
{
	mpz_t lambda;    /* lambda(n), i.e., lcm(p-1,q-1) */
	mpz_t x;   /* cached to avoid recomputing */
} paillier_prvkey_t;

/*
  This is a (semantic rather than structural) type for plaintexts.
  These can be converted to and from ASCII strings and byte arrays.
*/
typedef struct
{
	mpz_t m;
} paillier_plaintext_t;

/*
  This is a (semantic rather than structural) type for ciphertexts.
  These can also be converted to or from byte arrays (for example in
  order to store them in a file).
*/
typedef struct
{
	mpz_t c;
} paillier_ciphertext_t;

/*
  This is the type of the callback functions used to obtain the
  randomness needed by the probabilistic algorithms. The functions
  paillier_get_rand_devrandom and paillier_get_rand_devurandom
  (documented later) may be passed to any library function requiring a
  paillier_get_rand_t, or you may implement your own. If you implement
  your own such function, it should fill in "len" random bytes in the
  array "buf".
*/
typedef void (*paillier_get_rand_t) ( void* buf, int len );

/*****************
 BASIC OPERATIONS
*****************/

/*
  Generate a keypair of length modulusbits using randomness from the
  provided get_rand function. Space will be allocated for each of the
  keys, and the given pointers will be set to point to the new
  paillier_pubkey_t and paillier_prvkey_t structures. The functions
  paillier_get_rand_devrandom and paillier_get_rand_devurandom may be
  passed as the final argument.
*/
void paillier_keygen( int modulusbits,
			paillier_pubkey_t** pub,
											paillier_prvkey_t** prv,
											paillier_get_rand_t get_rand );

/*
	Encrypt the given plaintext with the given public key using
	randomness from get_rand for blinding. If res is not null, its
	contents will be overwritten with the result. Otherwise, a new
	paillier_ciphertext_t will be allocated and returned.
*/
 paillier_ciphertext_t* paillier_enc( paillier_ciphertext_t* res,
					 paillier_pubkey_t* pub,
					 paillier_plaintext_t* pt,
					 paillier_get_rand_t get_rand );

/*
	Decrypt the given ciphertext with the given key pair. If res is not
	null, its contents will be overwritten with the result. Otherwise, a
	new paillier_plaintext_t will be allocated and returned.
*/
paillier_plaintext_t* paillier_dec( paillier_plaintext_t* res,
																		paillier_pubkey_t* pub,
																		paillier_prvkey_t* prv,
																		paillier_ciphertext_t* ct );

/*****************************
 USE OF ADDITIVE HOMOMORPHISM
*****************************/

/*
	Multiply the two ciphertexts assuming the modulus in the given
	public key and store the result in the contents of res, which is
	assumed to have already been allocated.
*/
void paillier_mul( paillier_pubkey_t* pub,
									 paillier_ciphertext_t* res,
									 paillier_ciphertext_t* ct0,
									 paillier_ciphertext_t* ct1 );

/*
  Raise the given ciphertext to power pt and store the result in res,
  which is assumed to be already allocated. If ct is an encryption of
  x, then res will become an encryption of x * pt mod n, where n is
  the modulus in pub.
*/
void paillier_exp( paillier_pubkey_t* pub,
									 paillier_ciphertext_t* res,
									 paillier_ciphertext_t* ct,
									 paillier_plaintext_t* pt );

/****************************
 PLAINTEXT IMPORT AND EXPORT
****************************/

/*
  Allocate and initialize a paillier_plaintext_t from an unsigned long
  integer, an array of bytes, or a null terminated string.
*/
paillier_plaintext_t* paillier_plaintext_from_ui( unsigned long int x );
paillier_plaintext_t* paillier_plaintext_from_bytes( void* m, int len );
paillier_plaintext_t* paillier_plaintext_from_str( char* str );

/*
	Export a paillier_plaintext_t as a null terminated string or an
	array of bytes. In either case the result is allocated for the
	caller and the original paillier_plaintext_t is unchanged.
*/
char* paillier_plaintext_to_str( paillier_plaintext_t* pt );
void* paillier_plaintext_to_bytes( int len, paillier_plaintext_t* pt );

/*****************************
 CIPHERTEXT IMPORT AND EXPORT
*****************************/

/*
	Import or export a paillier_ciphertext_t from or to an array of
	bytes. These behave like the corresponding functions for
	paillier_plaintext_t's.
*/
paillier_ciphertext_t* paillier_ciphertext_from_bytes( void* c, int len );
void* paillier_ciphertext_to_bytes( int len, paillier_ciphertext_t* ct );

/**********************
 KEY IMPORT AND EXPORT
**********************/

/*
	Import or export public and private keys from or to hexadecimal,
	ASCII strings, which are suitable for I/O. Note that the
	corresponding public key is necessary to initialize a private key
	from a hex string. In all cases, the returned value is allocated for
	the caller and the values passed are unchanged.
*/
char* paillier_pubkey_to_hex( paillier_pubkey_t* pub );
char* paillier_prvkey_to_hex( paillier_prvkey_t* prv );
paillier_pubkey_t* paillier_pubkey_from_hex( char* str );
paillier_prvkey_t* paillier_prvkey_from_hex( char* str,
																						 paillier_pubkey_t* pub );

/********
 CLEANUP
********/

/*
  These free the structures allocated and returned by various
  functions within library and should be used when the structures are
  no longer needed.
*/
void paillier_freepubkey( paillier_pubkey_t* pub );
void paillier_freeprvkey( paillier_prvkey_t* prv );
void paillier_freeplaintext( paillier_plaintext_t* pt );
void paillier_freeciphertext( paillier_ciphertext_t* ct );

/***********
 MISC STUFF
***********/

/*
	These functions may be passed to the paillier_keygen and
	paillier_enc functions to provide a source of random numbers. The
	first reads bytes from /dev/random. On Linux, this device
	exclusively returns entropy gathered from environmental noise and
	therefore frequently blocks when not enough is available. The second
	returns bytes from /dev/urandom. On Linux, this device also returns
	environmental noise, but augments it with a pseudo-random number
	generator when not enough is available. The latter is probably the
	better choice unless you have a specific reason to believe it is
	insufficient.
*/
void paillier_get_rand_devrandom(  void* buf, int len );
void paillier_get_rand_devurandom( void* buf, int len );

/*
	This function just allocates and returns a paillier_ciphertext_t
	which is a valid, _unblinded_ encryption of zero (which may actually
	be done without knowledge of a public key). Note that this
	encryption is UNBLINDED, so don't use it unless you want anyone who
	sees it to know it is an encryption of zero. This function is
	sometimes handy to get some homomorphic computations started or
	quickly allocate a paillier_ciphertext_t in which to place some
	later result.
*/
paillier_ciphertext_t* paillier_create_enc_zero();

/*
	Just a utility used internally when we need round a number of bits
	up the number of bytes necessary to hold them.
*/
#define PAILLIER_BITS_TO_BYTES(n) ((n) % 8 ? (n) / 8 + 1 : (n) / 8)
