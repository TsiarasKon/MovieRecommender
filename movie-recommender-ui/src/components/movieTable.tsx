import React, { useEffect, useState } from 'react';
import { Button, Input, Rate, Space, Table, Tag } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import Movie from '../types';
import { sorterStringCompare } from '../utils';
import { FilterDropdownProps } from 'antd/lib/table/interface';
import { AllGenres } from '../constantData';
import '../styles/movieTable.css';

const MovieTableComponent = ({ allMovies, setParentMovies }: { allMovies: Movie[], setParentMovies: (newMovies: Movie[]) => void }) => {
  const defaultPageSize = 10

  const [movies, setMovies] = useState(allMovies);
  const [searchText, setSearchText] = useState('');  // eslint-disable-line @typescript-eslint/no-unused-vars

  useEffect(() => setMovies(allMovies), [allMovies]);

  const handleSearch = (selectedKeys: any, confirm: any): void => {
    confirm();
    setSearchText(selectedKeys[0]);
  }

  const handleReset = (clearFilters: any): void => {
    clearFilters();
    setSearchText('');
  }

  const columns = [
    {
      title: 'Title',
      dataIndex: 'primaryTitle',
      key: 'primaryTitle',
      ellipsis: true,
      sorter: (a: Movie, b: Movie) => sorterStringCompare(a.primaryTitle, b.primaryTitle),
      filterDropdown: ({ setSelectedKeys, selectedKeys, confirm, clearFilters }: FilterDropdownProps) => (
        <div style={{ padding: 8 }}>
          <Input
            // ref={node => {
            //   this.searchInput = node;
            // }}
            placeholder={`Search Title`}
            value={selectedKeys[0]}
            onChange={e => setSelectedKeys(e.target.value ? [e.target.value] : [])}
            onPressEnter={() => handleSearch(selectedKeys, confirm)}
            style={{ marginBottom: 8, display: 'block' }}
          />
          <Space>
            <Button
              type="primary"
              onClick={() => handleSearch(selectedKeys, confirm)}
              icon={<SearchOutlined />}
              size="small"
              style={{ width: 90 }}
            >
              Search
            </Button>
            <Button onClick={() => handleReset(clearFilters)} size="small" style={{ width: 90 }}>
              Reset
            </Button>
          </Space>
        </div>
      ),
      filterIcon: (filtered: boolean) => <SearchOutlined style={{ color: filtered ? '#1890ff' : undefined }} />,
      onFilter: (value: any, record: any) =>
        record.primaryTitle
          ? record.primaryTitle.toString().toLowerCase().includes(value.toLowerCase())
          : '',
      render: (t: string, m: Movie) =>
        <a href={'https://www.imdb.com/title/' + m.tconst} style={{ display: 'block' }} target={"_blank"} rel="noreferrer">{t}</a>
    },
    {
      title: 'Genres',
      dataIndex: 'genres',
      key: 'genres',
      render: (genres: any) => genres && genres.split(',').map((g: string) =>
        <Tag key={g} style={{ margin: 2 }}>
          {g}
        </Tag>
      ),
      filters: AllGenres.map((g: any) => ({
        text: <Tag key={g}>{g}</Tag>
        // {g}
        ,
        value: g,
      })),
      onFilter: (value: any, record: Movie) =>
        record.genres.includes(value)
    },
    {
      title: 'Year',
      dataIndex: 'startYear',
      key: 'startYear',
      sorter: (a: Movie, b: Movie) => a.startYear - b.startYear,
      width: 100,
      align: 'center' as 'center'
    },
    {
      title: 'IMDb Rating',
      dataIndex: 'averageRating',
      key: 'averageRating',
      sorter: (a: Movie, b: Movie) => a.averageRating - b.averageRating,
      width: 150,
      align: 'center' as 'center'
    },
    {
      title: 'Votes',
      dataIndex: 'numVotes',
      key: 'numVotes',
      sorter: (a: Movie, b: Movie) => a.numVotes - b.numVotes,
      width: 100,
      align: 'center' as 'center'
    },
    {
      title: 'Your Rating',
      dataIndex: 'userRating',
      key: 'userRating',
      sorter: (a: Movie, b: Movie) => (a.userRating ?? 0) - (b.userRating ?? 0),
      render: (userRating: number, record: Movie) =>
        <Rate allowHalf value={userRating} key={record.tconst} onChange={
          (value: number) => {
            const newMovies = movies.map(m => (m.tconst === record.tconst ? { ...m, userRating: value } : m));
            // setMovies(newMovies);    // this is unnecessary due to the useEffect hook above
            setParentMovies(newMovies);
          }
        } />,
      width: 200,
      align: 'center' as 'center'
    },
  ];

  return (
    <Table className="movie-table" dataSource={movies} columns={columns} pagination={{ defaultPageSize: defaultPageSize }} />
  );
}

export default MovieTableComponent;
