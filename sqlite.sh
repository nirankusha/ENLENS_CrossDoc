#!/bin/bash
set -e

PREFIX=$HOME/.local
PYVER=3.11.9

# --- Build ncurses ---
wget https://ftp.gnu.org/pub/gnu/ncurses/ncurses-6.5.tar.gz
tar xvf ncurses-6.5.tar.gz
cd ncurses-6.5
./configure --prefix=$PREFIX
make && make install
cd ..

# --- Build libffi ---
wget https://github.com/libffi/libffi/releases/download/v3.4.6/libffi-3.4.6.tar.gz
tar xvf libffi-3.4.6.tar.gz
cd libffi-3.4.6
./configure --prefix=$PREFIX
make && make install
cd ..

# --- Build readline ---
wget https://ftp.gnu.org/gnu/readline/readline-8.2.tar.gz
tar xvf readline-8.2.tar.gz
cd readline-8.2
./configure --prefix=$PREFIX
make && make install
cd ..

# --- Build SQLite (latest autoconf tarball) ---
wget https://www.sqlite.org/2025/sqlite-autoconf-3510000.tar.gz
tar xvf sqlite-autoconf-3510000.tar.gz
cd sqlite-autoconf-3510000
./configure --prefix=$PREFIX
make && make install
cd ..

# --- Rebuild Python with pyenv ---
pyenv uninstall -f $PYVER || true

CPPFLAGS="-I$PREFIX/include" \
LDFLAGS="-L$PREFIX/lib" \
PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig" \
LD_LIBRARY_PATH="$PREFIX/lib" \
CONFIGURE_OPTS="--enable-loadable-sqlite-extensions" \
pyenv install $PYVER

# --- Verify ---
echo "Testing Python build..."
~/.pyenv/versions/$PYVER/bin/python -c "import sqlite3, ctypes, curses, readline; print('SQLite:', sqlite3.sqlite_version, '| All modules OK')"
