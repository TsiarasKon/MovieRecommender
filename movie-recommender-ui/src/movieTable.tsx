import React, { useState } from 'react';
import { Button, Col, Input, Rate, Row, Space, Table, Tag } from 'antd';
import { SearchOutlined } from '@ant-design/icons';
import MoviesData from "./data/movies.json";
import Movie from './types';
import { loadJsonData, sorterStringCompare } from './utils';
import { FilterDropdownProps } from 'antd/lib/table/interface';
import { AllGenres } from './constantData';
import RecommendationsComponent from './recommendations';

function MovieTableComponent() {
  const AllMovies: Movie[] = JSON.parse(loadJsonData(MoviesData));
  const [movies, setMovies] = useState(AllMovies)
  const defaultPageSize = 10

  const [searchText, setSearchText] = useState('');  // eslint-disable-line @typescript-eslint/no-unused-vars

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
        record.genres.includes(value),
    },
    {
      title: 'Year',
      dataIndex: 'startYear',
      key: 'startYear',
      sorter: (a: Movie, b: Movie) => a.startYear - b.startYear
    },
    {
      title: 'Rating',
      dataIndex: 'averageRating',
      key: 'averageRating',
      sorter: (a: Movie, b: Movie) => a.averageRating - b.averageRating
    },
    {
      title: '# votes',
      dataIndex: 'numVotes',
      key: 'numVotes',
      sorter: (a: Movie, b: Movie) => a.numVotes - b.numVotes
    },
    {
      title: 'Your Rating',
      dataIndex: 'userRating',
      key: 'userRating',
      sorter: (a: Movie, b: Movie) => (a.userRating ?? 0) - (b.userRating ?? 0),
      render: (userRating: number, record: Movie) =>
        <Rate allowHalf value={userRating} key={record.tconst} onChange={
          (value: number) => setMovies(movies.map(m => (m.tconst === record.tconst ? { ...m, userRating: value } : m)))
        } />
    },
  ];

  return (
    <div>
      <Row justify="space-between">
        <Col span={17}>
          <Table dataSource={movies} columns={columns} pagination={{ defaultPageSize: defaultPageSize }} />
        </Col>
        <Col span={6}>
          <RecommendationsComponent ratedMovies={movies.filter(m => !!m.userRating)} allMovies={AllMovies} />
        </Col>
      </Row>
    </div>
  );
}

export default MovieTableComponent;
